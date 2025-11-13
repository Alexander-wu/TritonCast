
## TritonCast: Advanced Long-term Earth System Forecasting

## ✨ Core Highlights

- Exceptional Long-term Stability: Completes a full year of purely autoregressive forecasting.
- SOTA-level Forecast Accuracy: On par with leading AI and NWP models on WeatherBench 2.
- Unprecedented Zero-shot Generalization: Generalizes from coarse to fine resolutions without retraining.
- High-Fidelity Eddy Forecast: Extends effective forecast horizon for ocean eddies to 120 days.

## 🤖 Model Zoo

| Directory                              | Domain                  | Specific Task                        | Params          | Core Training Data                   | Notes / Key Features                                    |
| ------------------------------------------------ | ----------------------- | ------------------------------------ | --------------- | ------------------------------------ | ------------------------------------------------------- |
| `exp1_medium_range_weather_forecasting`          | Atmospheric Science     | Medium-range Weather                 | 1B              | ERA5 @ 1.5°, 6-hr steps              | **Heavyweight accuracy model**, for SOTA benchmark      |
| `exp2_long_term_stability_test`                  | Atmospheric Science     | Long-term Stability Test             | 0.02B           | ERA5 @ 1.0°, 24-hr steps             | **Lightweight stability model**, completes 365-day forecast |
| `exp3_multi_year_climate_simulation`             | Atmospheric Science     | Multi-year Climate Sim.              | 0.1B            | ERA5 + GLORYS12 @ 1.5°               | **Climate simulation model**, for long-term physical response |
| `exp4_global_ocean_simulation_and_forecasting`   | Oceanography            | Global Ocean Simulation              | 0.02B           | GLORYS12 + ERA5 @ 0.25°              | **Global ocean model**, validates coupled forecast robustness |
| `exp6_high_fidelity_eddy_forecast`               | Oceanography            | High-Fidelity Eddy Forecast          | 0.028B          | CMEMS @ 0.125° (3 regions)           | **High-res regional model**, for eddy-resolving forecasts |
| `exp6_high_fidelity_eddy_forecast`               | Oceanography            | Zero-shot Generalization             | 0.002B          | Global CMEMS @ 0.25° coarse-res      | **Ultra-lightweight model**, for cross-resolution capability |
| `exp7_isotropic_turbulence`                      | Theoretical Physics     | 2D Isotropic Turbulence              | 0.02B           | 128x128 DNS Data                     | **Physics-benchmark model**, for turbulence dynamics test |

## 🚀 Getting Started

### 1. Environment Setup

```bash
echo "--- Setting up environment ---"
git clone https://github.com/your-repo/TritonCast-Main-4.git
cd TritonCast-Main-4
conda create -n tritoncast python=3.9 -y
conda activate tritoncast
pip install -r requirements.txt
echo "--- Environment setup complete ---"
