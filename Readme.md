# TritonCast: Advanced Long-term Earth System Forecasting
*A deep learning framework for accurate and stable long-term Earth system modeling*

---

<div align="center">

[![Paper](https://img.shields.io/badge/paper-PDF-red?style=for-the-badge)](https://arxiv.org/abs/your_paper_id)
[![Hugging Face Models](https://img.shields.io/badge/Models-Hugging%20Face-blue?style=for-the-badge)](https://huggingface.co/TritonCast/TritonCast_model)
[![Hugging Face Datasets](https://img.shields.io/badge/Datasets-Hugging%20Face-yellow?style=for-the-badge)](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets)

</div>

> TritonCast is an innovative deep learning framework designed to fundamentally address the "spectral bias" problem in AI models. Through its unique **V-cycle multigrid architecture** and a **Latent Dynamical Core**, it achieves unprecedented long-term simulation stability while maintaining high short-term forecast accuracy.

<br>

<p align="center">
  <img src="TritonCast.png" alt="TritonCast Architecture and Key Results" width="90%">
  <br>
  <em><b>Figure 1:</b> (a) The V-cycle architecture; (b) Stable year-long atmospheric forecast; (c) Skill scores in ocean forecasting; (d) High-fidelity 120-day forecast of ocean eddies.</em>
</p>

---

## ✨ Core Highlights

-   ** exceptional Long-term Stability**: Completes a full year of purely autoregressive forecasting and successfully predicted the 2020 Siberian heatwave nearly six months in advance.
-   **🏆 SOTA-level Forecast Accuracy**: On the WeatherBench 2 benchmark, its performance is comparable to or exceeds that of other leading AI models and traditional NWP systems.
-   **🌐 Unprecedented Zero-shot Generalization**: A model trained on coarse-resolution data can directly generate physically realistic forecasts on unseen high-resolution grids.
-   **🌊 High-Fidelity Eddy Forecast**: Extends the effective forecast horizon for ocean mesoscale eddies from ~10 days to an unprecedented 120 days.

---

## 🤖 Model Zoo

A family of pretrained models is provided to cover all experiments presented in the paper.

| Directory (`exp*`)                               | Domain                  | Specific Task                        | Params          | Core Training Data                   | Notes / Key Features                                    |
| ------------------------------------------------ | ----------------------- | ------------------------------------ | --------------- | ------------------------------------ | ------------------------------------------------------- |
| `exp1_medium_range_weather_forecasting`          | Atmospheric Science     | Medium-range Weather                 | 1B              | ERA5 @ 1.5°, 6-hr steps              | **Heavyweight accuracy model**, for SOTA benchmark      |
| `exp2_long_term_stability_test`                  | Atmospheric Science     | Long-term Stability Test             | 0.02B           | ERA5 @ 1.0°, 24-hr steps             | **Lightweight stability model**, completes 365-day forecast |
| `exp3_multi_year_climate_simulation`             | Atmospheric Science     | Multi-year Climate Sim.              | 0.1B            | ERA5 + GLORYS12 @ 1.5°               | **Climate simulation model**, for long-term physical response |
| `exp4_global_ocean_simulation_and_forecasting`   | Oceanography            | Global Ocean Simulation              | 0.02B           | GLORYS12 + ERA5 @ 0.25°              | **Global ocean model**, validates coupled forecast robustness |
| `exp6_high_fidelity_eddy_forecast`               | Oceanography            | High-Fidelity Eddy Forecast          | 0.028B          | CMEMS @ 0.125° (3 regions)           | **High-res regional model**, for eddy-resolving forecasts |
| `exp6_high_fidelity_eddy_forecast`               | Oceanography            | Zero-shot Generalization             | 0.002B          | Global CMEMS @ 0.25° coarse-res      | **Ultra-lightweight model**, for cross-resolution capability |
| `exp7_isotropic_turbulence`                      | Theoretical Physics     | 2D Isotropic Turbulence              | 0.02B           | 128x128 DNS Data                     | **Physics-benchmark model**, for turbulence dynamics test |

---

## 🚀 Getting Started

### 1. Environment Setup

```bash
# Clone this repository
git clone https://github.com/your-repo/TritonCast-Main-4.git
cd TritonCast-Main-4

# Create and activate the conda environment (Python 3.8+ recommended)
conda create -n tritoncast python=3.9 -y
conda activate tritoncast

# Install dependencies
pip install -r requirements.txt
