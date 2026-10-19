# 1h Zarr data pipeline — design & plan of action

This document describes the refactor of `notebooks/data_to_tfrecords.ipynb`
into a reusable data pipeline that builds a **1h dataset** from NANUQ + ERA5,
stores it as **Zarr** instead of TFRecords, and makes the **temporal
averaging** of fields an explicit, configurable step.

## 1. Why a refactor

The notebook was a single, hard-coded, TF1/TFRecord pipeline that:

* rebuilt the ERA5→neXtSIM interpolation index map inside the per-year loop
  (wasteful — the grid does not change);
* wrote intermediate per-year NetCDF files, then merged and re-saved them,
  then wrote TFRecords — several copies of the same data;
* hard-coded the 6h `timestep_difference` and implicitly aligned times by
  reassigning coordinates;
* mixed the two ERA5 layouts (winds already on `gridT`, other vars on native
  lat/lon) so interpolation was inconsistent;
* stored raw, un-normalised fields separately from the TFRecords, requiring a
  fragile multi-step normalisation pass.

The refactor keeps the same physical content (NANUQ sea-ice state + 7 ERA5
forcings, optionally ocean) but organises it around a single **1h master
Zarr store** per split, with a clean separation between **grid**,
**temporal** and **normalisation** concerns.

## 2. Architecture

```
nainuq/datapipeline/
├── config.py          Config dataclass (paths, splits, variables, chunking)
├── grid.py            Regridder (ERA5 lat/lon -> neXtSIM 128x128) + pad/trim
├── temporal.py        Temporal averaging rules + increment computation
├── build.py           Orchestrates reading NANUQ/ERA5, regridding, writing Zarr
└── normalization.py   Mean/std over the training split
nainuq/datasets/ZarrDataset.py   PyTorch Dataset reading the Zarr stores
build_dataset.py       CLI entrypoint
```

Output layout (per split): a consolidated Zarr store

```
{save_dir}/train.zarr
{save_dir}/val.zarr
{save_dir}/test.zarr
{save_dir}/normalization.zarr
```

Every variable is stored at its **native 1h resolution**, on the 128×128 grid,
with `(time, y, x)` dims and `time`-chunked encoding so that lazy, chunked
reads stay cheap. The `prec` singleton dimension of the notebook is dropped.

## 3. Grid handling (correct interpolation)

All ERA5 variables — including the winds — are treated identically:

1. Read native hourly ERA5 on its regular `lat`/`lon` grid.
2. Interpolate to the neXtSIM curvilinear grid with **nearest-neighbour in
   3D Cartesian space** (great-circle-correct across the pole), via a single
   `Regridder` whose index map is built **once** and reused for every
   variable and every year (`grid.Regridder`).
3. Pad/trim onto the square 128×128 grid (`grid.pad_trim_to_grid`) exactly as
   the notebook did: pad x by `(5,5)`, trim y to 128.

This removes the notebook's inconsistent treatment of winds vs. other
forcings and avoids rebuilding the `cKDTree` inside the year loop.

## 4. Temporal averaging (the key design point)

The master store keeps **every field at 1h**. Training then chooses, per
variable and per target resolution, how to represent time. This is the
explicit decision the notebook made implicitly.

### The problem

The planned split between thermodynamics and dynamics imposes two different
temporal representations on the *same* underlying 1h data:

| Group | Variables | Target resolution | Forcing temporal representation |
|---|---|---|---|
| **Thermodynamic** | `sit`, `sic`, `snt` | **6h** increment | **6h-mean / 6h-accumulated** ERA5 |
| **Dynamical** | `siu`, `siv` | **1h** increment (cumulative) | **native 1h** ERA5 |

For a 6h thermodynamic step, feeding the *instantaneous* ERA5 forcing at the
start of the window loses the energy/mass that accumulates over the 6 hours
— for flux variables (`mtpr` precipitation, `msdswrf`/`msdlwrf` radiation)
the physically meaningful input is the **time-mean rate over the window**.
For dynamical (wind-driven) training at 1h, the native hourly forcing is what
drives the ice velocity.

### How the pipeline handles it

* The master store is **1h and lossless** — nothing is averaged away.
* `temporal.aggregate_time(ds, freq, rules)` resamples to a coarser
  resolution using a **per-variable aggregation rule**:
  * `mean`  — window mean (fluxes/rates: `mtpr`, `msdswrf`, `msdlwrf`; smooth
    fields: `t2m`, `d2m`, `u10`, `v10`),
  * `sum`   — window sum (future accumulated quantities, e.g. precipitation
    totals if ever needed),
  * `sample`— instantaneous (state variables `sit`…`snt`, ocean state).
* `temporal.compute_increments(state, steps)` builds the targets:
  * dynamical: `steps=1` → 1h increments (accumulated by the model during
    rollout = **cumulative learning**),
  * thermodynamic: `steps=6` → 6h increments with 6h-averaged forcings.
* `ZarrDataset` exposes this as `resolution` (`"1h"` / `"6h"`) and
  `cumulative` flags, so the same master store serves both the 6h
  thermodynamic model and the hourly cumulative dynamical model.

> **Design note / open decision:** for flux variables at 6h, choose between
> *mean rate* (keep in W/m² / kg/m²/s, matching the trained input space) or
> *accumulated total* (integrate over the window). The current default is
> **mean rate** to stay consistent with the existing model inputs; switching
> to accumulation only requires changing the rule for `mtpr`/`msdswrf`/
> `msdlwrf` to `sum`.

## 5. Normalisation

Mean and std of every stored variable are computed over the **training**
split and saved to `normalization.zarr`
(`normalization.build_normalization`). The `ZarrDataset` applies them on the
fly, replacing the notebook's separate `*_normalized.nc` pass.

## 6. Plan of action

**Phase 1 — Pipeline scaffolding (this branch, done)**
1. `Config` dataclass and CLI (`build_dataset.py`) replacing the argparse
   cells. ✅
2. `Regridder` + pad/trim (`grid.py`). ✅
3. Temporal averaging + increments (`temporal.py`). ✅
4. 1h Zarr build orchestration (`build.py`), writing consolidated
   `train/val/test.zarr` by yearly append. ✅
5. Normalisation (`normalization.py`) and `ZarrDataset` reader. ✅

**Phase 2 — Validate on a single year (needs cluster access)**
6. Run `build_dataset.py` on **one** year (e.g. 2010, or a single month via
   `--months_begin/--months_end`) and sanity-check the store: dims
   `(time, 128, 128)`, correct time alignment of NANUQ vs ERA5, regridded
   fields match the old notebook output on a few cells.
7. Confirm the ERA5 file patterns / NANUQ globs against the real directory
   layout (adjust `atmo_file_pattern`, `sim_dir` globs if the files are
   organised differently than assumed).
8. Compare a regridded field against the notebook's `interpolate_all_variables`
   output for the same variable/year to confirm the index map is identical.

**Phase 3 — Full build + normalisation**
9. Launch the full train (2010–2017), val (2018), test (2019–2020) build,
   then compute `normalization.zarr`. Verify sizes and timing.

**Phase 4 — Wire into training**
10. Add a `--dataset zarr` option to `train_emulator.py` that uses
    `ZarrDataset` instead of `Sea_ice_dataset` (same channel layout, so the
    UNet is unchanged).
11. Smoke-train one epoch on a tiny subset.

**Phase 5 — Thermodynamic vs dynamical split (later stage)**
12. Train a **thermodynamic** model (`variables=["sit","sic","snt"]`,
    `resolution="6h"`) and a **dynamical** model (`variables=["siu","siv"]`,
    `resolution="1h"`, `cumulative=True`) from the same master store.
13. Implement cumulative rollout in the dynamical model (accumulate 1h
    increments over 6 steps) and validate against the 6h reference.
14. Iterate on the temporal-averaging rule (mean rate vs accumulated flux) by
    comparing validation RMSE.

## 7. Dependencies

Build pipeline needs `xarray`, `zarr`, `numpy`, `scipy`, `pandas`
(present in `nainuq.yml`: `xarray 2023.1.0`, `zarr 2.16.1`). `ZarrDataset`
additionally needs `torch`. No TensorFlow is required for the new pipeline.
