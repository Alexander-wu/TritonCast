This repo is the official PyTorch implementation of Triton_Earth: **TritonCast: Advanced Long-term Earth System Forecasting**.

<p align="left">
<a href="https://arxiv.org/abs/2505.19432" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2306.11249-b31b1b.svg?style=flat" /></a>
<a href="https://github.com/easylearningscores/Triton_AI4Earth/blob/main/LICENSE" alt="license">
    <img src="https://img.shields.io/badge/license-Apache--2.0-%23002FA7" /></a>
</p>

[📘Documentation](https://tritoncast4earth.netlify.app/) |
[🛠️Installation](docs/en/install.md) |
[🚀Model Zoo](https://huggingface.co/TritonCast) |
[🤗Huggingface](https://huggingface.co/TritonCast) |
[👀Visualization](https://tritoncast4earth.netlify.app/) |
[🆕News](docs/en/changelog.md)



## 📑Open-source Plan

- [✅] [**Project Page**](https://tritoncast4earth.netlify.app/)
- [✅] [**Paper**](https://arxiv.org/abs/2505.19432)

## 🛠️Repository Structure
```
TritonCast-main/
├── exp1_medium_range_weather_forecasting/   # Corresponds to the medium-range weather forecasting experiments in the paper
├── exp2_long_term_stability_test/           # Corresponds to the long-term atmospheric stability experiments in the paper
├── exp3_multi_year_climate_simulation/      # Corresponds to the multi-year climate simulation experiments in the paper
├── exp4_global_ocean_simulation_and_forecasting/ # Corresponds to the global ocean simulation and forecasting experiments in the paper
├── exp6_high_fidelity_eddy_forecast/        # Corresponds to the high-fidelity ocean eddy forecasting experiments in the paper, including zero-shot
├── exp7_isotropic_turbulence/               # Corresponds to the turbulence benchmark tests in the paper
└── Readme.md                                # This document
```

Below is a guide to the experiments presented in our paper and their corresponding code directories.

| Experiment Description | Directory | Quick Start |
| :--- | :---: | :---: |
| **Medium-Range Weather Forecasting** (on WeatherBench 2) | [`./exp1_...`](./exp1_medium_range_weather_forecasting) | [**Instructions**](./exp1_medium_range_weather_forecasting/README.md) |
| **Long-Term Atmospheric Stability Test** (Year-long forecast) | [`./exp2_...`](./exp2_long_term_stability_test) | [**Instructions**](./exp2_long_term_stability_test/README.md) |
| **Multi-Year Climate Simulation** | [`./exp3_...`](./exp3_multi_year_climate_simulation) | [**Instructions**](./exp3_multi_year_climate_simulation/README.md) |
| **Global Ocean Simulation & Forecasting** | [`./exp4_...`](./exp4_global_ocean_simulation_and_forecasting) | [**Instructions**](./exp4_global_ocean_simulation_and_forecasting/README.md) |
| **High-Fidelity Ocean Eddy Forecast** | [`./exp6_...`](./exp6_high_fidelity_eddy_forecast) | [**Instructions**](https://github.com/Alexander-wu/TritonCast/blob/main/exp6_high_fidelity_eddy_forecast%20/Readme.md) |
| **Isotropic Turbulence Benchmark** | [`./exp7_...`](./exp7_isotropic_turbulence) | [**Instructions**](./exp7_isotropic_turbulence/README.md) |
## 🚀Architecture 

<div align="center">
  <img src="Figures/TritonCast.jpg" alt="TritonCast Architecture" width="1080">
</div>
Figure: The V-cycle architecture of TritonCast. It integrates a Multi-Grid Hierarchy for multi-scale processing, a stable Latent Dynamical Core (LDC) for long-term evolution, and Skip-Connections to retain high-fidelity details. This design effectively mitigates error accumulation in long-term forecasts.

## 🌟 Highlights

TritonCast establishes a new state-of-the-art in long-term Earth system forecasting. Our key contributions include:

-   **🌀 Unprecedented Long-term Stability**: Achieves stable, year-long, purely autoregressive global atmospheric forecasts without any drift or model collapse, accurately capturing seasonal cycles.
-   **🌊 High-Fidelity Ocean Forecasting**: Extends the skillful forecast of ocean eddies to an unprecedented 120 days, preserving fine-scale structures that other models lose.
-   **🏆 State-of-the-Art Performance**: Matches or exceeds leading AI models (like Pangu-Weather, GraphCast) and operational systems on the WeatherBench 2 benchmark for medium-range forecasting.
-   **🌐 Zero-Shot Generalization**: Demonstrates a remarkable ability to generalize across resolutions—a model trained on 0.25° data can produce physically realistic forecasts on unseen 0.125° grids, proving it has learned the underlying physical laws.
