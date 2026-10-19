"""Data pipeline to build NaiNUQ training datasets from NANUQ + ERA5 into Zarr.

Replaces the monolithic ``data_to_tfrecords`` notebook:

* **Grid handling** is done once via a reusable nearest-neighbour regridder
  (ERA5 regular lat/lon -> neXtSIM curvilinear 128x128 Arctic grid).
* **Temporal handling** is explicit: the pipeline builds a canonical **1h**
  master store and exposes a temporal-aggregation layer so that
  thermodynamic variables can later be trained at 6h while dynamical
  variables are trained hourly with cumulative learning.
* **Storage** is Zarr instead of TFRecords.

Submodules
----------
config        : Config dataclass (paths, splits, variables, chunking).
grid          : Regridder + pad/trim helpers.
temporal      : Temporal averaging rules + increment computation.
build         : Orchestrates reading NANUQ/ERA5, regridding and writing Zarr.
normalization : Compute/save mean & std over the training split.
"""

from .config import Config, Splits
from .grid import Regridder, pad_trim_to_grid
from .temporal import DEFAULT_AGG_RULES, aggregate_time, compute_increments

__all__ = [
    "Config",
    "Splits",
    "Regridder",
    "pad_trim_to_grid",
    "DEFAULT_AGG_RULES",
    "aggregate_time",
    "compute_increments",
]
