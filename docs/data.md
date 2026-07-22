# Data

This page describes the data pipeline for NaiNUQ: how raw NANUQ outputs
are preprocessed into TFRecord shards, what variables are included, and
how the dataset is structured for training and evaluation. It also includes 
information on how to download 

---

## Overview

NaiNUQ learns to emulate one-step increments of sea-ice state variables
from a combination of the current sea-ice state and atmospheric (and
optionally ocean) forcings. All data is defined on a curvilinear **Arctic 1°
grid (128×128 cells)**.

!!! note
    The land/ocean mask is stored separately in `src/mask2_nanuk1.npy`
    and applied during post-processing. Grid cells covered by land are
    set to zero.

---

## Data download
NANUK simulation can be found on this [storage cloud](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/simus_nanuq/NANUK1/old/NANUK1-N1CPL00-S/catalog.html)

Note that in the folder there is several simulations from NANUK available:

- `NANUK1-CPL00-S` : 2010-2020 simulation (2009 is used as one year of spin-up). In this simulation, only the surface ocean is saved. 
- `NANUK1-N1CPL00-S` : 2010-2020 simulation (2009 is used as one year of spin-up). In this simulation, the velocities of the first layer of the ocean is solved under two name `vozocrtx`, `vomecrty`

For both simulations, the simulation output are saved as yearly **netcdf files** with one repertory per year.

In each folder, several files are available 
- `{config_name}_1h_{year}0101_{year}1231_icemod.nc` : hourly output of ocean - sea ice simulation 
- `{config_name}_6h_{year}0101_{year}1231_icemod.nc` : 6h-hourly averaged output of ocean - sea ice simulation
- `{config_name}_6h_{year}0101_{year}1231_ovel30.nc` : For the `NANUK1-N1CPL00-S`, first layer of the ocean surface

Additionaly, all information about the grid is contained in this [file](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/MEOM/simus_nanuq/NANUK1/old/NANUK1-N1CPL00-S/catalog.html?dataset=meomscanpublic/MEOM/simus_nanuq/NANUK1/old/NANUK1-N1CPL00-S/mesh_mask_NANUK1_L75_4.2.2.nc)


## Variables

### Sea-ice state variables

These are the prognostic variables emulated by NaiNUQ. The model predicts
their **increment** (change over one time step), which is added to the
current state autoregressively.

| Short name | Description | Units |
|---|---|---|
| `sit` | Sea-ice thickness | m |
| `sic` | Sea-ice concentration | — (0–1) |
| `siu` | Zonal ice velocity | m/s |
| `siv` | Meridional ice velocity | m/s |
| `snt` | Snow thickness | m |

The emulator can be trained on any subset of these variables. The
`--sea_ice_variables` argument controls which subset is used (see
[Inference](inference.md)).


### Atmospheric forcings (7 channels)


|                  Forcings name                 | Abbreviations |      Units      |
|------------------------------------------------|---------------|-----------------|
| 10 meter $u$-velocity                          |       U10     |       m/s       |
| 10 meter $v$-velocity                          |       V10     |       m/s       |
| 2 meter temperature                            |       T2M     |        K        |
| Mean total precipitation rate                  |      mtpr     |  kg m^-2^ s^-1^ |
| Mean surface direct short-wave radiation flux  |    msdrswrf   |      W m^-2     |
| Mean surface downward long-wave radiation flux |    msdwlwrf   |      W m^-2     |
|  2 meter dewpoint temperature                  |       D2M     |        K        |
|------------------------------------------------|---------------|-----------------|


### Ocean forcings (optional, 5 channels)

Ocean surface forcings can be included as additional input channels by
setting `--use_ocean_as_forcings True`. This increases `N_tot_input`
from 12 to 17.

|         Forcings name          | Abbreviations | Units |
|--------------------------------|---------------|-------|
| Sea surface temperature        |       SST     |   K   |
| Sea surface height             |       SSH     |   m   |
| Sea surface salinity           |       SSS     |  psu  |
| Sea surface/bulk $u$-velocity  |       SSU     |  m/s  |
| Sea surface/bulk $v$-velocity  |       SSV     |  m/s  | 
|--------------------------------|---------------|-------|

---

## Input tensor layout

Each input tensor has shape `(C_in, 128, 128)`. The channel ordering is:

```
Channels 0–4   : sea-ice state variables (sit, sic, siu, siv, snt)
Channels 5–11  : atmospheric forcings (7 variables)
Channels 12–13 : ocean under-ice (optional, N_under=2)
Channels 14–18 : ocean surface forcings (optional, N_ocean=5)
```

`C_in` is therefore **12** (no ocean) or **17** (with ocean).

The output tensor has shape `(C_out, 128, 128)` where `C_out` is the
number of emulated variables (1, 2, or 5).

---

## Normalization

All variables are normalized to zero mean and unit variance before
being fed to the model. Normalization statistics are precomputed over
the training set and saved as `.npy` files in the data directory:

| File | Description |
|---|---|
| `sea_ice_mean_input.npy` | Per-channel input mean (sea ice) |
| `sea_ice_std_input.npy` | Per-channel input std (sea ice) |
| `sea_ice_mean_output.npy` | Per-channel output mean (sea ice) |
| `sea_ice_std_output.npy` | Per-channel output std (sea ice) |
| `ocean_mean_input.npy` | Per-channel input mean (ocean, optional) |
| `ocean_std_input.npy` | Per-channel input std (ocean, optional) |
| `ocean_mean_output.npy` | Per-channel output mean (ocean, optional) |
| `ocean_std_output.npy` | Per-channel output std (ocean, optional) |

---

## TFRecord format

Data is stored as **TFRecord shards** (TensorFlow's binary sequential
format) for efficient I/O. Each shard contains a sequence of records,
each with two fields:

```
inputs  : float32 flat array of shape (C_in × 128 × 128,)
outputs : float32 flat array of shape (5 × 128 × 128,)
```

Shards follow the naming convention:

```
data_{year}_{forcing}.tfrecords.{shard_id:03d}
```

For example: `data_2018_jra.tfrecords.000`.

!!! note
    Two atmospheric forcing datasets are currently supported: **JRA-55**
    (`jra`) and **TOPAZ** (`topaz`). The corresponding SLURM scripts
    (`jra.slurm`, `topaz.slurm`) handle job submission for each.

---

## Dataset splits

| Split | Years | Location |
|---|---|---|
| Training | TODO | `{data_path}/train/` |
| Validation | TODO | `{data_path}/val/` |
| Test | 2019, 2020 | `{data_path}/test/` |

---

## Preprocessing pipeline

!!! warning "TODO"
    Describe the steps to go from raw NANUQ NetCDF outputs to TFRecord
    shards (interpolation, normalization, sharding). Link to the
    preprocessing scripts if available.

---

## Loading the dataset

The [`Sea_ice_dataset`](api/datasets.md) class wraps the TFRecord shards
as a standard PyTorch `Dataset` and is used with a `DataLoader`:

```python
from datasets.TFRecordDataset import Sea_ice_dataset
from torch.utils.data import DataLoader

files = [f"data/test/data_2019_jra.tfrecords.{str(i).zfill(3)}" for i in range(N_files)]

dataset = Sea_ice_dataset(
    filenames=files,
    variables=["sit", "sic", "siu", "siv", "snt"],
    N_ocean=0,
    N_under=0,
    use_ocean=False,
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)
```
