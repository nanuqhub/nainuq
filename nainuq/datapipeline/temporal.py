"""Temporal averaging and increment computation.

This is the piece the original notebook handled implicitly and that the new
1h dataset makes explicit. The master store keeps every field at its native
**1h** resolution. Training then needs to decide, per variable and per target
resolution, which temporal representation to feed the network.

Design
------
* **Thermodynamical variables** (``sit``, ``sic``, ``snt`` and the flux-like
  forcings) are planned to be trained at **6h**. Their target is the 6h
  increment and the relevant forcing is the **6h-mean / 6h-accumulated**
  ERA5 field, because fluxes (radiation, precipitation) integrate energy and
  mass over the window.
* **Dynamical variables** (``siu``, ``siv``) are planned to be trained
  **hourly with cumulative learning**: the model predicts 1h increments and
  accumulates them over the 6 steps. Here the forcing is the **native 1h**
  field.

The aggregation rule is therefore a per-variable choice. :data:`DEFAULT_AGG_RULES`
provides sensible defaults; ``aggregate_time`` applies a rule map over
non-overlapping windows, and ``compute_increments`` builds the targets.
"""

from __future__ import annotations

from typing import Dict

import xarray as xr

# Default temporal-aggregation rule per variable when resampling to a coarser
# resolution (e.g. 1h -> 6h):
#   "mean"    : window mean  (fluxes/rates and smooth fields)
#   "sum"     : window sum   (accumulated quantities)
#   "sample"  : keep first/instantaneous value (states)
DEFAULT_AGG_RULES: Dict[str, str] = {
    # ---- ERA5 atmospheric forcings ----
    "t2m": "mean",   # 2m temperature: mean over window
    "d2m": "mean",   # 2m dew point:   mean over window
    "u10": "mean",   # 10m wind:       mean over window
    "v10": "mean",   # 10m wind:       mean over window
    "mtpr": "mean",  # mean total precip rate: keep as a mean *rate*
    "msdswrf": "mean",  # downward short-wave flux (W/m2): mean rate
    "msdlwrf": "mean",  # downward long-wave flux  (W/m2): mean rate
    # ---- NANUQ state variables (keep instantaneous) ----
    "sit": "sample",
    "sic": "sample",
    "siu": "sample",
    "siv": "sample",
    "snt": "sample",
    # ---- ocean ----
    "zos": "sample",
    "tos": "sample",
    "sos": "sample",
    "vozocrtx": "mean",
    "vomecrty": "mean",
}


def aggregate_time(
    ds: xr.Dataset,
    freq: str,
    rules: Dict[str, str] | None = None,
) -> xr.Dataset:
    """Resample a 1h dataset to ``freq`` over non-overlapping windows.

    Parameters
    ----------
    ds : xr.Dataset
        1h dataset with a ``time`` coordinate.
    freq : str
        Target frequency, e.g. ``"6h"``. Windows are left-closed, labelled
        by their left edge (so a window covering 00-06Z is labelled 00Z).
    rules : dict
        Per-variable aggregation rule. Defaults to :data:`DEFAULT_AGG_RULES`.

    Returns
    -------
    xr.Dataset
        Aggregated dataset with a ``time`` coordinate at ``freq``.
    """
    rules = rules or DEFAULT_AGG_RULES
    # Left-closed, non-overlapping window labels, e.g. 00Z for [00,06).
    bucket = ds["time"].dt.floor(freq)
    bucket.name = "time"
    out = {}
    for name, da in ds.data_vars.items():
        rule = rules.get(name, "mean")
        grouped = da.groupby(bucket)
        if rule == "sum":
            out[name] = grouped.sum()
        elif rule == "sample":
            out[name] = grouped.first()
        else:
            out[name] = grouped.mean()
    return xr.Dataset(out)


def compute_increments(state: xr.Dataset, steps: int = 1) -> xr.Dataset:
    """State increment over ``steps`` time steps: ``state[t+steps] - state[t]``.

    The last ``steps`` timesteps are dropped (no future state to subtract).
    """
    diff = state.shift(time=-steps) - state
    return diff.isel(time=slice(0, -steps))
