# NaiNUQ × CMIP — Subproject Specification

> **Status:** Draft — to be reviewed before starting any implementation work.
> **Scope chosen:** **Option A** — use a CMIP6 model's *atmospheric* fields as forcings,
> keeping NANUQ/NANUK1 ocean fields and sea-ice state as targets.

---

## 1. Goal

Adapt the NaiNUQ data + inference pipeline so the emulator is driven by **CMIP6 atmospheric
forcings** instead of **ERA5**, while reusing the existing model weights, inference code, and
targets (NANUQ/NANUK1 sea-ice state) unchanged.

- First milestone: **run the emulator in inference** on a CMIP-forced dataset and compare against
  the ERA5-forced baseline.
- Second milestone (only if inference degrades): **retrain** the emulator on CMIP-forced TFRecords.

## 2. Background — the current pipeline (as built in this repo)

Data flow today (see `notebooks/data_to_tfrecords.ipynb` and `docs/data.md`):

1. **Atmospheric forcings** (ERA5, 7 channels) regridded to the Arctic 1° 128×128 curvilinear grid:

   | Repo var | Description | CMIP6 equivalent |
   |---|---|---|
   | `t2m`   | 2 m temperature        | `tas`  |
   | `d2m`   | 2 m dewpoint temp      | `hurs` (→ derive dewpoint) or `tas`+`huss` |
   | `mtpr`  | Mean total precip rate | `pr`   |
   | `msdlwrf`| Surface downwelling longwave | `rlds` |
   | `msdswrf`| Surface downwelling shortwave | `rsds` |
   | `u10` / `v10` | 10 m wind components | `uas` / `vas` |

2. **Ocean forcings** (5 channels, optional) from the coupled **NANUK1** run (which was itself
   forced by ERA5): `tos` (SST), `sos` (SSS), `zos` (SSH), `vozocrtx`/`vomecrty` (surface ocean
   velocities). **Option A keeps these from NANUK1 — only the atmosphere is replaced.**

3. **Targets** = NANUQ/NANUK1 sea-ice state (`sit`, `sic`, `siu`, `siv`, `snt`).

4. Normalization constants (`sea_ice_mean_input.npy`, `ocean_*.npy`, …) → TFRecord shards →
   `Sea_ice_dataset` → inference / training.

`C_in = 17` (5 sea-ice + 7 atmosphere + 5 ocean) for this configuration.

## 3. What "Option A" means concretely

| Component | Before (ERA5) | After (CMIP) |
|---|---|---|
| Atmospheric forcings (7 ch) | ERA5 reanalysis | CMIP6 model (chosen) |
| Ocean forcings (5 ch) | NANUK1 (ERA5-forced) | **unchanged** (NANUK1) |
| Sea-ice targets | NANUQ/NANUK1 | **unchanged** |
| Model weights | `weights/last.ckpt` | **unchanged** (inference first) |
| Grid | Arctic 1° 128×128 curvilinear | same — CMIP regridded onto it |

**Key implication:** because only the atmosphere changes, the 5 sea-ice + 5 ocean channels are
already in the right grid/units. Only the **7 atmospheric channels** must be sourced from CMIP and
regridded onto the curvilinear grid. This dramatically narrows the regridding and normalization
scope compared to Option B.

## 4. Open decisions to lock before implementation

1. **Scenario & period.** Which CMIP6 experiments / time range?
   - `historical` (typically 1850–2014) vs `ssp*` (2015–2100).
   - Must the CMIP period **overlap** the NANUK1/ERA5 target years (2009–2020)? If the target is
     inference on 2019–2020, we need a CMIP model whose `historical`/`ssp` covers those years.
   - Decision needed: use historical-era CMIP for 2009–2020, or accept CMIP forcing outside the
     trained distribution and measure the distribution shift.
2. **Model selection.** Which CMIP6 model(s)? Criteria:
   - Arctic sea-ice skill (SIPN / CMIP6 sea-ice intercomparison literature).
   - Availability of all 7 atmospheric variables at a usable resolution (ideally ≥ ~1°; many are
     1–2°).
   - Temporal coverage matching the chosen period.
3. **Atmosphere-only regridding** target is already fixed (curvilinear 1° grid); the only open
   question is interpolation method (nearest-neighbour via the existing `cKDTree`, vs bilinear).
4. **Normalization.** Whether to recompute atmospheric normalization over the CMIP forcing
   distribution (likely yes — otherwise inputs fall outside the trained input distribution).

## 5. Variable mapping (CMIP → repo), atmospheric channels only

| CMIP6 | Standard name | Repo input channel | Notes |
|---|---|---|---|
| `tas`  | air_temperature | `t2m` | K |
| `hurs` | relative_humidity | `d2m` | convert RH + `tas` → dewpoint |
| `pr`   | precipitation_flux | `mtpr` | kg m⁻² s⁻¹ |
| `rlds` | surface_downwelling_longwave_flux_in_air | `msdlwrf` | W m⁻² |
| `rsds` | surface_downwelling_shortwave_flux_in_air | `msdswrf` | W m⁻² |
| `uas`  | eastward_wind | `u10` | m s⁻¹ |
| `vas`  | northward_wind | `v10` | m s⁻¹ |

## 6. Build order / workstreams

### Phase 0 — Mission spec & selection (research)
- [ ] Review this SPEC, confirm decisions in §4.
- [ ] **Model-selection research report** — shortlist of CMIP6 models with Arctic skill +
      variable availability + period coverage + resolution. Deliverable: `cmip/model_selection.md`.
- [ ] Finalize the **variable mapping** table (§5) and any unit/convention conversions.
- [ ] Decide regridding method + normalization strategy.

### Phase 1 — Data acquisition
- [ ] Download chosen CMIP6 atmospheric variables (Arctic domain, relevant period) from ESGF.
- [ ] Stage data in a `cmip/data/` or configured path.
- [ ] Document exact file layout & naming.

### Phase 2 — Regridding + TFRecord rebuild
- [ ] Adapt `data_to_tfrecords.ipynb` → parameterized script that reads CMIP atmosphere +
      NANUK1 ocean/targets.
- [ ] Regrid 7 atmospheric channels onto Arctic 1° curvilinear grid (reuse existing `cKDTree`).
- [ ] Recompute normalization constants (sea-ice unchanged; atmospheric over CMIP distribution).
- [ ] Regenerate TFRecord shards; verify `Sea_ice_dataset` loads them.

### Phase 3 — Inference
- [ ] Run inference with the existing two-year pipeline (reuse `nainuq_two_year_forecast.ipynb`
      logic) on the CMIP-derived shards.
- [ ] Compare RMSE / bias vs the ERA5 baseline; quantify distribution shift.

### Phase 4 — Retraining (conditional)
- [ ] Only if inference metrics degrade beyond a threshold.
- [ ] Retrain emulator on CMIP TFRecords via `train_emulator.py` + a CMIP SLURM script.
- [ ] Re-evaluate on a held-out CMIP period.

## 7. Success criteria

- **Primary:** CMIP-forced inference runs end-to-end and produces comparable RMSE/bias to the
  ERA5 baseline (target: within a documented, agreed margin; absolute parity is not expected because
  CMIP forcing is not a reanalysis).
- **Secondary:** a written assessment of the distribution shift between ERA5 and CMIP forcing for
  the 7 atmospheric channels.

## 8. Risks & mitigations

| Risk | Mitigation |
|---|---|
| CMIP grid much coarser than 1° → regridding artefacts | Model selection prioritizes resolution; validate regridded fields visually & statistically |
| CMIP forcing outside trained distribution → poor inference | Measure shift; Option A keeps ocean/targets fixed so shift is bounded to atmosphere; retrain if needed |
| Missing variables in chosen model (e.g. no `rlds`/`rsds`) | Filter shortlist by full 7-var availability |
| Dewpoint conversion introduces error | Document conversion; consider using a model that directly provides what's needed |
| Large downloads / storage | Download Arctic-subset only; clear storage budget |

## 9. Open questions for the team

1. Target period for inference (2019–2020 matching the current test set)?
2. Acceptable RMSE margin vs ERA5 baseline?
3. Storage/bandwidth budget for CMIP downloads?
4. Any preferred CMIP6 model already in mind?
