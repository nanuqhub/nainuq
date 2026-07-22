# NaiNUQ

**NaiNUQ** is a deep learning emulator of [NANUQ](https://github.com/nanuqhub/nanuq), a numerical sea-ice model for the Arctic basin. It reproduces NANUQ outputs at a fraction of the computational cost, making it suitable for ensemble experiments, uncertainty quantification, and data-assimilation workflows.

!!! note "Status"
    This project is under active development. API and configuration formats may change between versions. See the [CHANGELOG](https://github.com/cdurand95/nanuq1/blob/main/CHANGELOG.md) for recent updates.

---

## Key features

| Feature | Details |
|---|---|
| **Domain** | Arctic basin, 1° spatial resolution |
| **Temporal resolutions** | 1 h, 6 h, 12 h, 24 h |
| **Architecture** | U-Net with partial convolutions (PConv) |
| **Framework** | TensorFlow / Keras |
| **HPC support** | SLURM job scripts included |

---

## Repository structure

```
nanuq1/
├── src/
│   ├── datasets/
│   │   └── TFRecordDataset.py     # TFRecord data pipeline
│   ├── layers/
│   │   ├── ModuleBlocks.py        # Reusable building blocks
│   │   ├── Pconv.py               # Partial convolution layer
│   │   ├── UNet.py                # Base U-Net architecture
│   │   ├── full_UNet.py           # Full U-Net model
│   │   └── full_UNet_with_PConv.py  # U-Net with partial convolutions
│   ├── inference/
│   │   ├── test.py                # Main inference script
│   │   ├── compute_metrics.py     # Evaluation metrics
│   │   ├── test_jra.py            # Inference with JRA forcing
│   │   ├── test_topaz.py          # Inference with TOPAZ forcing
│   │   └── *.slurm / *.sh         # HPC job scripts
│   └── train_emulator.py          # Training entry point
├── docs/                          # This documentation
├── CHANGELOG.md
└── README.md
```

---

## Quick start

### Installation

```bash
git clone https://github.com/cdurand95/nanuq1.git
cd nanuq1
pip install -r requirements.txt
```

### Training

```bash
python src/train_emulator.py --config configs/default.yaml
```

### Inference

```bash
python src/inference/test.py \
    --resolution 6h \
    --input data/input_state.nc \
    --output data/output_prediction.nc
```

!!! tip
    For HPC environments, use the provided SLURM scripts in `src/inference/`. See [Inference](inference.md) for details.

---

## How it works

NaiNUQ uses a **U-Net architecture with partial convolutions** to emulate the NANUQ sea-ice dynamics. Partial convolutions are key here: they allow the model to handle the irregular Arctic domain mask (land, coastlines) natively, rather than treating masked cells as zeros.

The emulator is trained to predict sea-ice state variables one time step ahead, and can be rolled out autoregressively for multi-step forecasts.

For a detailed description of the architecture, see [Architecture](architecture.md).

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

Please also cite the original NANUQ model:

```bibtex
@misc{nanuq,
  title        = {{NANUQ}: A sea-ice model for the Arctic basin},
  howpublished = {\url{https://github.com/nanuqhub/nanuq}},
}
```

---

## Contact

**Charlotte Durand** — charlotte.durand1@univ-grenoble-alpes.fr

This work is carried out within the [SASIP](https://github.com/sasip-climate) project.
