"""Orchestrates the construction of the 1h NaiNUQ Zarr dataset.

Pipeline (per split: train / val / test)
----------------------------------------
1. Open the NANUQ 1h ``icemod`` outputs, subset to the split years/months and
   extract the sea-ice state variables (plus ocean surface variables if used).
2. Open the NANUQ 1h ``ovel30`` outputs for the under-ice ocean forcings.
3. Open the hourly ERA5 forcing files, regrid them onto the neXtSIM grid with
   a single cached :class:`Regridder`, and align them on the NANUQ time axis.
4. Pad/trim every field onto the 128x128 grid and write the merged 1h master
   store as Zarr, appending year by year.

The resulting store is the lossless **1h master** dataset. Downstream
temporal aggregation (for 6h thermodynamic / 1h cumulative dynamical
training) is performed by :mod:`nainuq.datapipeline.temporal`.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

from .config import Config
from .grid import Regridder, build_target_grid, pad_trim_to_grid


# Map GRIB code (used in the ERA5 filename) -> variable name (used inside the file).
# Filenames look like ``ERA5_hourly_var167_y2010.nc`` while the netCDF variable
# inside is named ``t2m``.
ERA5_CODE_TO_NAME = {
    "var167": "t2m",
    "var165": "u10",
    "var166": "v10",
    "var168": "d2m",
    "var169": "msdswrf",
    "var172": "mtpr",
    "var175": "msdlwrf",
}
ERA5_NAME_TO_CODE = {name: code for code, name in ERA5_CODE_TO_NAME.items()}


def _era5_file_code(var: str) -> str:
    """Return the GRIB code used in the ERA5 filename for a logical variable."""
    return ERA5_NAME_TO_CODE.get(var, var)


# --------------------------------------------------------------------------- #
# NANUQ readers
# --------------------------------------------------------------------------- #
def _glob_year(pattern: str, year: int) -> List[str]:
    return sorted(glob.glob(pattern.replace("{year}", str(year))))


def open_nanuq_icemod(cfg: Config, year: int) -> xr.Dataset:
    """Open the 1h NANUQ sea-ice/ocean output files for a single year."""
    pattern = os.path.join(cfg.sim_dir, "**", f"*_1h_*{year}*_icemod.nc4")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No NANUQ icemod files for {year} in {cfg.sim_dir}")
    return xr.open_mfdataset(files, combine="by_coords")


def open_nanuq_ovel30(cfg: Config, year: int) -> xr.Dataset:
    """Open the 1h NANUQ under-ice ocean velocity files for a single year."""
    pattern = os.path.join(cfg.sim_dir, "**", f"*_1h_*{year}*_ovel30.nc4")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No NANUQ ovel30 files for {year} in {cfg.sim_dir}")
    return xr.open_mfdataset(files, combine="by_coords")


def get_nanuq_mesh(cfg: Config, year: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the neXtSIM ``nav_lat``/``nav_lon`` mesh from a NANUQ file."""
    ds = open_nanuq_icemod(cfg, year)
    lat = ds["nav_lat"].values
    lon = ds["nav_lon"].values
    ds.close()
    return lat, lon


# --------------------------------------------------------------------------- #
# ERA5 reader / regridder
# --------------------------------------------------------------------------- #
def open_era5(cfg: Config, variables: List[str], year: int) -> xr.Dataset:
    """Open hourly ERA5 files for a list of variables for one year and merge."""
    parts = []
    for var in variables:
        pattern = os.path.join(cfg.atmo_forcings_dir, cfg.atmo_file_pattern)
        pattern = pattern.replace("{var}", _era5_file_code(var)).replace("{year}", str(year))
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No ERA5 {var} files for {year} matching {pattern}")
        parts.append(xr.open_mfdataset(files, combine="by_coords"))
    merged = xr.merge(parts)
    return merged


def get_era5_lat_lon(cfg: Config, year: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the native ERA5 latitude/longitude (used to build the map)."""
    ds = open_era5(cfg, cfg.atmo_variables_all(), year)
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()
    return lat, lon


# --------------------------------------------------------------------------- #
# Sub-setting helpers
# --------------------------------------------------------------------------- #
def _select_years_months(ds: xr.Dataset, years, cfg: Config, time_dim: str):
    ds = ds.sel({time_dim: ds[time_dim].dt.year.isin(years)})
    ds = ds.sel({time_dim: ds[time_dim].dt.month.isin(range(cfg.months_begin, cfg.months_end + 1))})
    return ds


def _regrid_variables(
    src: xr.Dataset, variables: List[str], regridder: Regridder, time_dim: str
) -> xr.Dataset:
    """Regrid a set of (lat, lon) variables onto the target grid and rename the
    time dim to the canonical ``time``."""
    out = {}
    for var in variables:
        da = src[var]
        if time_dim != "time":
            da = da.rename({time_dim: "time"})
        out[var] = regridder.regrid(da)
    return xr.Dataset(out)


# --------------------------------------------------------------------------- #
# Main build
# --------------------------------------------------------------------------- #
def build_split(
    split: str,
    cfg: Config,
    regridder: Regridder,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> None:
    """Build (or append) the 1h Zarr store for one split."""
    years = list(cfg.splits.years(split))
    store_path = cfg.store_path(split)
    os.makedirs(cfg.save_dir, exist_ok=True)

    first = True
    for year in years:
        print(f"[{split}] building {year}")
        ds = _build_year(split, year, cfg, regridder, target_lat, target_lon)

        if first:
            ds.to_zarr(
                store_path,
                mode="w",
                consolidated=cfg.consolidated,
                encoding=_zarr_encoding(cfg, ds),
            )
            first = False
        else:
            ds.to_zarr(
                store_path,
                mode="a",
                append_dim="time",
                consolidated=cfg.consolidated,
                encoding=_zarr_encoding(cfg, ds),
            )
        ds.close()
    print(f"[{split}] wrote {store_path}")


def _build_year(
    split: str,
    year: int,
    cfg: Config,
    regridder: Regridder,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> xr.Dataset:
    months = range(cfg.months_begin, cfg.months_end + 1)
    years_sel = cfg.splits.years(split)

    # ---- NANUQ sea-ice state (+ ocean surface) ---------------------------
    nanuk = open_nanuq_icemod(cfg, year)
    nanuk = _select_years_months(nanuk, years_sel, cfg, cfg.nanuq_time_dim)
    state_vars = list(cfg.sea_ice_variables)
    if cfg.use_ocean:
        state_vars += [v for v in cfg.ocean_variables if v in nanuk]
    state = nanuk[state_vars].rename({cfg.nanuq_time_dim: "time"})

    # ---- NANUQ under-ice ocean forcings ----------------------------------
    under = None
    if cfg.use_ocean and cfg.ocean_forcings:
        ovel = open_nanuq_ovel30(cfg, year)
        ovel = _select_years_months(ovel, years_sel, cfg, cfg.nanuq_time_dim)
        under = ovel[cfg.ocean_forcings].isel(
            {cfg.nanuq_ovel_depth_dim: 0}
        ).rename({cfg.nanuq_time_dim: "time"})

    # ---- ERA5 atmospheric forcings, regridded ----------------------------
    era5 = open_era5(cfg, cfg.atmo_variables_all(), year)
    era5 = _select_years_months(era5, years_sel, cfg, "time")
    atmo = _regrid_variables(era5, cfg.atmo_variables_all(), regridder, "time")

    # ---- align every source on the canonical NANUQ time axis -------------
    master_time = state["time"]
    atmo = atmo.reindex(time=master_time, method="nearest", tolerance="1h")
    atmo["time"] = master_time
    if under is not None:
        under = under.reindex(time=master_time, method="nearest", tolerance="1h")
        under["time"] = master_time

    # ---- merge and pad to 128x128 ----------------------------------------
    master = xr.merge([state, atmo] + ([under] if under is not None else []))
    padded = _pad_master(master, cfg)
    padded = padded.assign_coords(
        latitude=(("y", "x"), target_lat),
        longitude=(("y", "x"), target_lon),
    )
    return padded


def _pad_master(master: xr.Dataset, cfg: Config) -> xr.Dataset:
    """Pad/trim every ``(time, y, x)`` variable onto the square grid."""
    out = {}
    for name, da in master.data_vars.items():
        if da.dims == ("time", "y", "x"):
            out[name] = pad_trim_to_grid(da, cfg.grid_size, cfg.pad_x)
        else:
            out[name] = da
    return xr.Dataset(out)


def _zarr_encoding(cfg: Config, ds: xr.Dataset) -> dict:
    """Per-variable Zarr chunking (time-chunked, full y/x per chunk)."""
    encoding = {}
    for name, da in ds.data_vars.items():
        if "time" in da.dims:
            encoding[name] = {
                "chunks": (min(cfg.time_chunk, da.sizes["time"]),) + (cfg.grid_size,) * 2
            }
    return encoding


def build_regridder(cfg: Config, anchor_year: int) -> Tuple[Regridder, np.ndarray, np.ndarray]:
    """Build the shared regridder + target mesh from anchor files."""
    tlat, tlon = get_nanuq_mesh(cfg, anchor_year)
    slat, slon = get_era5_lat_lon(cfg, anchor_year)
    regridder = Regridder(tlat, tlon, slat, slon)
    plat, plon = build_target_grid(tlat, tlon, cfg.grid_size, cfg.pad_x)
    return regridder, plat, plon


def build_all(cfg: Config) -> None:
    """Build the train / val / test stores."""
    anchor_year = cfg.splits.train[0]
    regridder, plat, plon = build_regridder(cfg, anchor_year)
    for split in ("train", "val", "test"):
        build_split(split, cfg, regridder, plat, plon)
