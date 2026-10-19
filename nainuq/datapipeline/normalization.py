"""Normalization statistics for the 1h Zarr dataset.

The mean and standard deviation of every stored variable are computed over
the **training** split and saved to a small Zarr store so that the model can
be fed zero-mean/unit-variance fields (matching the behaviour of the old
``*_normalized.nc`` files).
"""

from __future__ import annotations

import os

import xarray as xr

from .config import Config


def compute_normalization(cfg: Config) -> xr.Dataset:
    """Compute per-variable mean/std over the training store.

    Returns
    -------
    xr.Dataset
        With ``mean`` and ``std`` data variables, each shaped like the per-
        variable spatial field (y, x) with ``variable`` as the leading dim.
    """
    train = xr.open_zarr(cfg.store_path("train"), consolidated=cfg.consolidated)
    mean = train.mean(dim="time")
    std = train.std(dim="time")
    stats = xr.Dataset(
        {
            "mean": (("variable", "y", "x"), mean.to_array().data),
            "std": (("variable", "y", "x"), std.to_array().data),
        },
        coords={"variable": list(train.data_vars)},
    )
    train.close()
    return stats


def save_normalization(cfg: Config, stats: xr.Dataset) -> str:
    """Persist the normalization statistics to ``{save_dir}/normalization.zarr``."""
    path = os.path.join(cfg.save_dir, "normalization.zarr")
    stats.to_zarr(path, mode="w", consolidated=cfg.consolidated)
    print(f"Saved normalization stats to {path}")
    return path


def load_normalization(cfg: Config) -> xr.Dataset:
    """Load the saved normalization statistics."""
    return xr.open_zarr(
        os.path.join(cfg.save_dir, "normalization.zarr"), consolidated=cfg.consolidated
    )


def build_normalization(cfg: Config) -> None:
    """Compute and save normalization statistics for the training split."""
    save_normalization(cfg, compute_normalization(cfg))
