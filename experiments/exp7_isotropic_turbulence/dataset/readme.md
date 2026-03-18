# Isotropic Turbulence Dataset

This directory stores the evaluation data used by `exp7_isotropic_turbulence`, which reproduces the isotropic turbulence benchmark reported in the paper.

## Required File

Place the following file in this directory:

- `McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt`

Download source:

- [Hugging Face: scaomath/navier-stokes-dataset](https://huggingface.co/datasets/scaomath/navier-stokes-dataset/blob/main/McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt)

## Expected Directory Layout

```text
exp7_isotropic_turbulence/
├── dataset/
│   ├── readme.md
│   └── McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt
├── checkpoints/
├── config.yaml
└── inference.py
```

## Usage in This Repository

The default inference pipeline expects the dataset at:

```text
dataset/McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt
```

The file is loaded by `inference.py` and used to construct the held-out test split for model evaluation.

## Reproducibility Notes

- Please keep the original filename unchanged.
- Do not rename or repackage the `.pt` file; the current scripts assume the standard file path above.
- Make sure sufficient local disk space is available before downloading, as the dataset is large.
