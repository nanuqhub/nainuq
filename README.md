[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<br />
<div align="center">
  <a href="https://github.com/cdurand95/nanuq1">
    <img src="logo_nainuq.png" alt="Logo" width="300" height="300">
  </a>

  <h3 align="center">NaiNUQ</h3>

  <p align="center">
    A deep learning emulator of the <a href="https://github.com/nanuqhub/nanuq">NANUQ</a> sea-ice model
    <br />
    <a href="https://github.com/cdurand95/nanuq1"><strong>Explore the docs »</strong></a>
    &nbsp;·&nbsp;
    <a href="https://github.com/cdurand95/nanuq1/issues">Report Bug</a>
    &nbsp;·&nbsp;
    <a href="https://github.com/cdurand95/nanuq1/issues">Request Feature</a>
  </p>
</div>

---

## Table of Contents

- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Contact & Acknowledgments](#contact--acknowledgments)

---

## About

**NaiNUQ** is a deep-learning-based emulator of [NANUQ](https://github.com/nanuqhub/nanuq), a numerical sea-ice model operating on the **Arctic basin** at **1° spatial resolution**.

The emulator is designed to reproduce NANUQ outputs at a fraction of the computational cost, and is available at **four temporal resolutions**:

| Temporal resolution | Output frequency |
|---|---|
| 1 h | Hourly |
| 6 h | Sub-daily |
| 12 h | Twice daily |
| 24 h | Daily |

> **Why emulate?** Running full numerical sea-ice simulations is computationally expensive. NaiNUQ provides a fast surrogate suitable for ensemble experiments, uncertainty quantification, and data-assimilation workflows. 

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

Make sure the following are installed before proceeding:

- Python ≥ 3.9
- `pip` or `conda`
- Inference runs on **CPU, CUDA GPU, or Apple Silicon (MPS)**. A CUDA GPU or an MPS device gives
  near-real-time inference, but the code also runs on CPU. The two-year forecast only needs ~a few
  hundred MB of RAM because data is streamed from disk (no 40 GB GPU/RAM requirement).



### Installation

1. Clone the repository:

```bash
git clone https://github.com/nanuqhub/nainuq.git
cd nainuq
```

2. Create a conda environment. Two files are provided:

- `nainuq.yml` — Linux with a CUDA-capable GPU (original training/inference setup):

```bash
conda env create -f nainuq.yml
conda activate nainuq
```

- `nainuq_arm.yml` — macOS on Apple Silicon (MPS), used to run `nainuq_two_year_forecast.ipynb`:

```bash
conda env create -f nainuq_arm.yml
conda activate nainuq_arm
```

> On Apple Silicon the `nainuq` environment can crash (SIGABRT) while reading TFRecords; use
> `nainuq_arm` for the notebooks. You can create a kernel for Jupyter with:
> `python -m ipykernel install --user --name nainuq_arm`

3. Download pre-trained model weights available on Hugging Face [here](https://huggingface.co/cdurand95/nainuq) and place them in the `weights/` directory:

```
huggingface-cli download cdurand95/nainuq --local-dir weights

```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Demo

### Downloading subset of test dataset

A zenodo capsule hold a subset of the 2019 test dataset for users to easily play with the emulator, in the 6h temporal resolution and with sub-surface ocean currents to match the weights of the emulator saved on HuggingFace.
It is available [here](https://doi.org/10.5281/zenodo.23102333)
The dataset can be download with the command 
```bash
cd data
zenodo_get 23102334
```

It is saved as TFRecords (see the [documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=fr)) and they are used as inputs of the neural network relying the function TFRecordDataset.py

### Downloading the two-year test dataset

The **full two-year test set (2019 and 2020)** used by `nainuq_two_year_forecast.ipynb` is available on
[Zenodo](https://doi.org/10.5281/zenodo.23102333). Place the TFRecord shards (`data_2019.tfrecords.*`
and `data_2020.tfrecords.*`) together with the normalisation constants (`sea_ice_*.npy`) in the `data/`
directory:

```bash
cd data
zenodo_get 23102334
```

### Running inference

To test the emulator, a demo notebook is available under `notebooks/nainuq_demo_6h_rmse.ipynb` to
reproduce 6h sub-surface emulator 10 day forecast on the test years. A second notebook,
`notebooks/nainuq_two_year_forecast.ipynb`, runs the emulator over the **full two years** (2019+2020,
1440 forecasts every 12 h), reports the **NN inference timing**, and is **memory-efficient**: it
streams the TFRecords from disk instead of loading them all, so it runs on a laptop with 16 GB RAM and
does not save the prediction fields (only the aggregate RMSE / bias / persistence metric arrays in
`notebooks/two_year_output/`).

To run the two-year notebook from the command line:

```bash
conda activate nainuq_arm
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 \
    notebooks/nainuq_two_year_forecast.ipynb
```

Otherwise, it can be run on a cluster using the slurm script `./inference/test_nainuq.slurm` or directly using the Python script `./inference/test.py`. The different arguments of the test script are described below
```bash
# Run inference
python inference/test.py \
    --save_dir $SAVE_DIR \ #Directory path where the weights of the NN are saved
    --data_path $DATA_PATH \ #Directory path where the dataset are stored
    --NN_size $NN_size \ #NN initial number of channels for 1st block (32 by default)
    --frequency $FREQUENCY \ #Frequency between two trajectory starts (default=12 for short term forecast - meaning bi-daily forecast)
    --pconv_use $PCONV_USE \ #Whether partial convolution (Liu 2018) are used in the NN (default=0)
    --post_processing $POST_PROCESSING \ #Whether or not post-processing rules are applied (default=1)
    --checkpoint_name $CHECKPOINT_NAME \ #Name under which the model is saved
    --n_cycle $N_CYCLE \ #Number of trajectory (default = 1200 for short term forecast)
    --ocean_variables $OCEAN_VARIABLES\ #Whether or not ocean variables are predicted (default=0)
    --ocean_under $OCEAN_UNDER\ #Unused
    --use_ocean_as_forcings $use_ocean_as_forcings\ #Whether or not ocean variables are used as forcings (default=1)
    --sea_ice_variables $SEA_ICE_VARIABLES \ #Which sea ice variables are predicted (default=["sit","sic","siu","siv","snt"])
    --ocean $OCEAN \ #Unused
    --k $K \ #Number of iteration of the NN (typically set to 40 for 10 days forecast at 6h resolution)
    --timestep $TIMESTEP \ #Temporal resolution of the emulator
    --save_pred $SAVE_PRED \ #Whether or not each forecast is saved. If not, only RMSE and bias metrics are saved.
    --noise $NOISE \ #Flag to put noise in inputs (unused)
    --noise_init $NOISE_INIT #Std of noise if noise is activated
```

**Key arguments:**

| Argument | Description | Options |
|---|---|---|
| `--resolution` | Temporal resolution of the emulator | `1h`, `6h`, `12h`, `24h` |
| `--input` | Path to input NetCDF file (initial state) | — |
| `--output` | Path to write the predicted output | — |
| `--config` | Path to YAML configuration file | — |

### Input format

Input files should be NetCDF (`.nc`) files on the Arctic 1° grid containing the required sea-ice state variables. See `docs/input_format.md` for the full variable list and expected units.

### Output format

Outputs are written as NetCDF files matching the NANUQ variable naming conventions, making them directly comparable with reference model outputs.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Citation

If you use NaiNUQ in your research, please cite:

```bibtex
@article{durand2024nainuq,
  author  = {Durand, Charlotte and others},
  title   = {{NaiNUQ}: A deep learning emulator of the {NANUQ} sea-ice model},
  journal = {Journal TBD},
  year    = {2024},
  doi     = {10.XXXX/XXXXXX}
}
```

> ⚠️ *Citation information will be updated upon publication. Please check back or contact the author for the most recent reference.*

You may also want to cite the original NANUQ model:

```bibtex
@misc{nanuq,
  title        = {{NANUQ}: A sea-ice model for the Arctic basin},
  howpublished = {\url{https://github.com/nanuqhub/nanuq}},
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contact & Acknowledgments

**Charlotte Durand** — charlotte.durand1@univ-grenoble-alpes.fr

Project repository: [https://github.com/sasip-climate](https://github.com/sasip-climate)

This work is carried out within the [SASIP](https://github.com/sasip-climate) project. The authors acknowledge the developers of the NANUQ model for making their code openly available.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]: https://img.shields.io/github/contributors/nanuqhub/nainuq.svg?style=for-the-badge
[contributors-url]: https://github.com/nanuqhub/nainuq/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/nanuqhub/nainuq.svg?style=for-the-badge
[forks-url]: https://github.com/nanuqhub/nainuq/network/members
[stars-shield]: https://img.shields.io/github/stars/nanuqhub/nainuq.svg?style=for-the-badge
[stars-url]: https://github.com/nanuqhub/nainuq/stargazers
[issues-shield]: https://img.shields.io/github/issues/nanuqhub/nainuq.svg?style=for-the-badge
[issues-url]: https://github.com/nanuqhub/nainuq/issues
[license-shield]: https://img.shields.io/github/license/nanuqhub/nainuq.svg?style=for-the-badge
[license-url]: https://github.com/nanuqhub/nainuq/blob/master/LICENSE.txt
