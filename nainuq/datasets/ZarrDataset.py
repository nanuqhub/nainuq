"""PyTorch dataset reading the 1h NaiNUQ Zarr store.

This is the Zarr replacement for :class:`nainuq.datasets.TFRecordDataset`.
It opens the master 1h Zarr store, assembles inputs (state + forcings at each
sample time), computes the increment targets and normalises the fields, then
returns ``(inputs, outputs)`` tensors on the ``(C, 128, 128)`` layout the
UNet expects.

The design supports the planned split between thermodynamic and dynamical
training:

* **Thermodynamic** variables (``sit``, ``sic``, ``snt``): ``resolution="6h"``,
  ``cumulative=False``. Samples every 6h; the target is the 6h increment and
  the forcing is the 6h-mean field over the window.
* **Dynamical** variables (``siu``, ``siv``): ``resolution="1h"``,
  ``cumulative=True``. Samples every hour; the target is the native 1h
  increment which the model accumulates during rollout (cumulative learning).

Channel layout (before variable selection) matches the TFRecord convention:
``[state vars][atmospheric forcings][ocean under-ice][ocean surface]``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from ..datapipeline.temporal import compute_increments


def _steps_for(resolution: str) -> int:
    """Number of 1h master steps in the requested resolution."""
    return {"1h": 1, "6h": 6, "12h": 12, "24h": 24}[resolution]


def _window_mean(da, window: int):
    """Mean of ``da`` over the future window ``[t, t+window)``.

    Uses a trailing rolling mean shifted forward so the value at time ``t``
    is the mean of ``t .. t+window-1``. Trailing NaNs (incomplete windows)
    are ignored because only valid sample times are selected downstream.
    """
    if window == 1:
        return da
    return da.rolling(time=window, min_periods=window).mean().shift(time=-(window - 1))


class ZarrDataset(Dataset):
    """PyTorch ``Dataset`` over a NaiNUQ Zarr store.

    Parameters
    ----------
    store_path : str
        Path to a ``*.zarr`` store produced by
        :func:`nainuq.datapipeline.build.build_split`.
    variables : list of str
        Sea-ice variables to emulate (subset of ``{sit, sic, siu, siv, snt}``).
    resolution : str
        Temporal resolution of the targets, ``"1h"`` or ``"6h"``.
    cumulative : bool
        If ``True`` (dynamical learning), use native 1h increments sampled
        hourly and keep 1h forcings; otherwise use ``resolution``-spaced
        increments with ``resolution``-averaged forcings.
    normalization : optional xr.Dataset
        Mean/std statistics from
        :func:`nainuq.datapipeline.normalization.load_normalization`.
    """

    def __init__(
        self,
        store_path: str,
        variables: List[str],
        resolution: str = "6h",
        cumulative: bool = False,
        normalization=None,
    ):
        import xarray as xr

        self.variables = list(variables)
        self.resolution = resolution
        self.cumulative = cumulative
        self.normalization = normalization

        master = xr.open_zarr(store_path, consolidated=True)
        self.grid = (int(master.sizes["y"]), int(master.sizes["x"]))

        state = master[self.variables]
        self.forcing_vars = [v for v in _FORCING_ORDER if v in master]
        forcings = master[self.forcing_vars]

        # Sample geometry.
        res_h = 1 if cumulative else _steps_for(resolution)
        step = 1 if cumulative else res_h  # spacing between training samples

        n = state.sizes["time"]
        starts = np.arange(0, n - res_h, step)

        # Targets: positional increment over res_h steps (shift-based, so it
        # is NOT cancelled by xarray label alignment).
        increments = compute_increments(state, steps=res_h)  # times 0..n-res_h-1
        targets = increments.isel(time=starts)

        # Inputs: state at sample time + forcings averaged over [t, t+res_h).
        state_start = state.isel(time=starts)
        forcings_in = _window_mean(forcings, res_h).isel(time=starts)
        inputs = xr.merge([state_start, forcings_in])

        master.close()

        self.x = self._to_numpy(inputs, self.variables + self.forcing_vars)
        self.t = self._to_numpy(targets, self.variables)
        self.time = targets["time"].values

    # ------------------------------------------------------------------ #
    def _to_numpy(self, ds, variables):
        """Materialise ``(C, T, 128, 128)`` float32 array, normalised."""
        arr = (
            ds[variables]
            .to_array()
            .transpose("variable", "time", "y", "x")
            .values.astype(np.float32)
        )
        if self.normalization is not None:
            norm = self.normalization
            vars_in_norm = set(norm.coords.get("variable", []).values.tolist())
            for i, var in enumerate(variables):
                if var in vars_in_norm:
                    mean = norm["mean"].sel(variable=var).values
                    std = norm["std"].sel(variable=var).values
                    arr[i] = (arr[i] - mean) / np.where(std > 0, std, 1.0)
        return np.nan_to_num(arr, nan=0.0)

    # ------------------------------------------------------------------ #
    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        inputs = torch.from_numpy(self.x[:, idx]).float()
        outputs = torch.from_numpy(self.t[:, idx]).float()
        return inputs, outputs


# Order used to assemble the input channel vector (matches TFRecord layout).
_FORCING_ORDER = [
    "t2m", "d2m", "u10", "v10", "mtpr", "msdswrf", "msdlwrf",
    "vozocrtx", "vomecrty", "zos", "tos", "sos",
]
