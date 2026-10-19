"""NaiNUQ dataset loaders.

* :class:`TFRecordDataset.Sea_ice_dataset` : legacy TFRecord-based reader.
* :class:`ZarrDataset.ZarrDataset` : Zarr-based reader for the 1h dataset.
"""

from .ZarrDataset import ZarrDataset

__all__ = ["ZarrDataset"]
