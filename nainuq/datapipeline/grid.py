"""Grid regridding helpers for the NaiNUQ pipeline.

ERA5 fields are natively defined on a regular latitude/longitude grid while
NANUQ operates on a curvilinear Arctic 1 degree grid. This module provides a
:class:`Regridder` that interpolates ERA5 onto the neXtSIM grid using
nearest-neighbour search in 3D Cartesian space (so distances are great-circle
correct across the pole), plus a helper to pad/trim fields onto the square
128x128 grid the model expects.

The index map between the two grids is built **once** and reused for every
variable and every year, which is the main speed-up over the notebook where
the tree was rebuilt inside the per-year loop.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree


def lon_lat_to_cartesian(
    lon: np.ndarray, lat: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert longitude/latitude (degrees) to 3D Cartesian on a WGS-84
    ellipsoid. Used so that nearest-neighbour distances are geodesically
    meaningful, especially near the pole."""
    A = 6378.137  # major axis [km]
    E2 = 6.69437999014e-3  # eccentricity squared

    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    r_n = A / np.sqrt(1.0 - E2 * (np.sin(lat_rad) ** 2))
    x = r_n * np.cos(lat_rad) * np.cos(lon_rad)
    y = r_n * np.cos(lat_rad) * np.sin(lon_rad)
    z = r_n * (1.0 - E2) * np.sin(lat_rad)
    return x, y, z


class Regridder:
    """Nearest-neighbour regridder from a source lat/lon grid to a
    curvilinear target grid.

    Parameters
    ----------
    target_lat, target_lon : array-like (N_y, N_x)
        2D latitude/longitude of the target (e.g. neXtSIM ``nav_lat`` /
        ``nav_lon``).
    source_lat, source_lon : array-like
        Source grid. 1D regular coordinates are expanded with ``meshgrid``;
        2D grids (e.g. already-interpolated fields) are used as-is.
    """

    def __init__(self, target_lat, target_lon, source_lat, source_lon):
        slat, slon = np.asarray(source_lat), np.asarray(source_lon)
        if slat.ndim == 1 and slon.ndim == 1:
            slon2d, slat2d = np.meshgrid(slon, slat)
        else:
            slon2d, slat2d = slon, slat

        xs, ys, zs = lon_lat_to_cartesian(slon2d.ravel(), slat2d.ravel())
        self._tree = cKDTree(np.column_stack([xs, ys, zs]))

        tlat, tlon = np.asarray(target_lat), np.asarray(target_lon)
        xt, yt, zt = lon_lat_to_cartesian(tlon.ravel(), tlat.ravel())
        self._inds = self._tree.query(np.column_stack([xt, yt, zt]), k=1)[1]

        self.target_shape = np.shape(tlat)
        self.target_lat = tlat
        self.target_lon = tlon

    def regrid(self, da: xr.DataArray) -> xr.DataArray:
        """Regrid a DataArray whose last two dims are ``('lat', 'lon')``.

        Returns a DataArray with the same leading dims but ``('y', 'x')`` as
        the last two dims, on the target grid.
        """
        if da.dims[-2:] != ("lat", "lon"):
            raise ValueError(
                f"Expected trailing dims ('lat','lon'), got {da.dims[-2:]}"
            )
        lead_dims = da.dims[:-2]
        lead_shape = da.shape[:-2]
        flat = da.values.reshape(lead_shape + (-1,))
        out = flat[..., self._inds].reshape(lead_shape + self.target_shape)
        coords = {d: da.coords[d] for d in lead_dims}
        return xr.DataArray(out, dims=lead_dims + ("y", "x"), coords=coords)

    def target_lat_lon(self) -> Tuple[np.ndarray, np.ndarray]:
        """Latitude/longitude of the (unpadded) target grid."""
        return self.target_lat, self.target_lon


def pad_trim_to_grid(
    da: xr.DataArray,
    size: int = 128,
    pad_x: Tuple[int, int] = (5, 5),
) -> xr.DataArray:
    """Pad/trim a ``(y, x)`` field onto a square ``size x size`` grid.

    Mirrors the original notebook: pad x with ``pad_x`` zeros on each side,
    then keep the first ``size`` rows of y.
    """
    if "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"Expected ('y','x') dims, got {da.dims}")
    da = da.pad({"x": pad_x}, mode="constant", constant_values=0)
    da = da.isel(y=slice(0, size))
    return da


def build_target_grid(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    size: int = 128,
    pad_x: Tuple[int, int] = (5, 5),
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the padded/trimmed 128x128 target latitude/longitude arrays that
    are stored alongside the regridded fields."""
    lat = xr.DataArray(np.asarray(target_lat), dims=("y", "x"))
    lon = xr.DataArray(np.asarray(target_lon), dims=("y", "x"))
    return pad_trim_to_grid(lat, size, pad_x).values, pad_trim_to_grid(
        lon, size, pad_x
    ).values
