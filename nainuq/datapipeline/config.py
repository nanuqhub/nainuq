"""Configuration for the NaiNUQ 1h Zarr data pipeline.

A single :class:`Config` holds everything that used to be scattered across
``argparse`` calls in the ``data_to_tfrecords`` notebook: data locations,
temporal splits, the variable lists and the Zarr store layout.

Example
-------
>>> cfg = Config(
...     sim_dir="/path/to/NANUK1-N1CPL00-S",
...     atmo_forcings_dir="/path/to/ERA5",
...     save_dir="/path/to/output",
... )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Splits:
    """Year ranges used to build the train / val / test Zarr stores."""

    train: Tuple[int, int] = (2010, 2017)
    val: Tuple[int, int] = (2018, 2018)
    test: Tuple[int, int] = (2019, 2020)

    def years(self, split: str) -> range:
        b, e = getattr(self, split)
        return range(b, e + 1)


@dataclass
class Config:
    """Paths, splits, variables and store layout for the pipeline."""

    # --- paths -----------------------------------------------------------
    sim_dir: str = ""
    """Directory containing the NANUQ yearly netCDF outputs."""

    atmo_forcings_dir: str = ""
    """Directory containing the hourly ERA5 netCDF forcing files."""

    save_dir: str = ""
    """Root output directory (train.zarr / val.zarr / test.zarr live here)."""

    # --- temporal --------------------------------------------------------
    splits: Splits = field(default_factory=Splits)

    months_begin: int = 1
    months_end: int = 12

    # Resolution of the canonical master store (raw, lossless).
    master_freq: str = "1h"

    # --- grid ------------------------------------------------------------
    # Target square grid the regridded fields are padded/trimmed onto.
    grid_size: int = 128
    # Pad applied along x before trimming y (matches the original notebook:
    # native 118x129 -> pad x by (5,5) -> trim y to 128 -> 128x128).
    pad_x: Tuple[int, int] = (5, 5)

    # --- variables -------------------------------------------------------
    sea_ice_variables: List[str] = field(
        default_factory=lambda: ["sit", "sic", "siu", "siv", "snt"]
    )

    # Native ERA5 names (hourly). msdswrf / msdlwrf are the correct ERA5 ids.
    atmo_variables: List[str] = field(
        default_factory=lambda: ["t2m", "mtpr", "d2m", "msdswrf", "msdlwrf"]
    )
    wind_variables: List[str] = field(
        default_factory=lambda: ["u10", "v10"]
    )

    ocean_variables: List[str] = field(
        default_factory=lambda: ["zos", "tos", "sos"]
    )
    ocean_forcings: List[str] = field(
        default_factory=lambda: ["vozocrtx", "vomecrty"]
    )

    use_ocean: bool = True
    """Include ocean surface + under-ice fields in the store."""

    # --- ERA5 file layout -----------------------------------------------
    # glob patterns, with ``{var}`` and ``{year}`` placeholders, used to
    # locate hourly ERA5 files. All variables (including winds) are read from
    # the native ERA5 lat/lon grid and regridded consistently.
    atmo_file_pattern: str = "ERA5_hourly_{var}_y{year}.nc"

    # --- Zarr layout -----------------------------------------------------
    # Number of time steps per Zarr chunk (default ~1 month at 1h).
    time_chunk: int = 720
    consolidated: bool = True

    # --- NANUQ netCDF internals -----------------------------------------
    nanuq_time_dim: str = "time_counter"
    nanuq_ovel_depth_dim: str = "deptht30"

    def atmo_variables_all(self) -> List[str]:
        """All atmospheric forcing variables (thermo + wind)."""
        return list(self.atmo_variables) + list(self.wind_variables)

    def store_path(self, split: str) -> str:
        return f"{self.save_dir}/{split}.zarr"
