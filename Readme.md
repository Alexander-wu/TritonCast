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

- [✅] Project Page
- [✅] Github Page
- [✅] Paper

## 🚀Architecture 

<div align="center">
  <img src="Figures/TritonCast.jpg" alt="TritonCast Architecture" width="1080">
</div>
Figure: The V-cycle architecture of TritonCast. It integrates a Multi-Grid Hierarchy for multi-scale processing, a stable Latent Dynamical Core (LDC) for long-term evolution, and Skip-Connections to retain high-fidelity details. This design effectively mitigates error accumulation in long-term forecasts.

## 🌟 Highlights

TritonCast establishes a new state-of-the-art in long-term Earth system forecasting. Our key contributions include:

🌀 **Unprecedented Long-term Stability:** Achieves stable, year-long, purely autoregressive global atmospheric forecasts without any drift or model collapse, accurately capturing seasonal cycles.

🌊 **High-Fidelity Ocean Forecasting:** Extends the skillful forecast of ocean eddies to an unprecedented 120 days, preserving fine-scale structures that other models lose.

🏆 **State-of-the-Art Performance:** Matches or exceeds leading AI models (like Pangu-Weather, GraphCast) and operational systems on the WeatherBench 2 benchmark for medium-range forecasting.

🌐 **Zero-Shot Generalization:** Demonstrates a remarkable ability to generalize across resolutions—a model trained on 0.25° data can produce physically realistic forecasts on unseen 0.125° grids, proving it has learned the underlying physical laws.
