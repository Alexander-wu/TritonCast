# High-Fidelity Ocean Eddy Forecasting (exp6)

This document provides instructions for reproducing the inference results of the **High-Fidelity Ocean Eddy Forecasting** experiments from the TritonCast paper.

This experiment module is divided into two main parts:
1.  **Global 0.25° Forecasts**: Includes standard inference, baseline comparisons, and the zero-shot generalization test.
2.  **Regional 0.125° Forecasts**: High-resolution forecasts for specific eddy-rich regions (Kuroshio, Gulf Stream, Agulhas Current).

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have already activated the main conda environment for this project. If not, please follow the setup instructions in the root `README.md`.

```bash
conda activate triton_v2
```

### 2. Download Data

Download the required CMEMS dataset for this experiment and place it in a designated `data` directory.

- **Download Link**: [**CMEMS_dataset**](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/CMEMS_dataset) and [**zero-shot test datasets**](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Zero-shot%20Generalization)

We recommend the following structure:
```
exp6_high_fidelity_eddy_forecast/
├── data/
│ └── CMEMS_dataset/
│ ├── test/
│ └── test_0125/
```


### 3. Download Pre-trained Models

Download the pre-trained model weights for this experiment from Hugging Face.

- **Download Link**: [**exp6_high_fidelity_eddy_forecast weights**](https://huggingface.co/TritonCast/TritonCast_model/tree/main/exp_high_fidelity_eddy_forecast%20)

Place the downloaded model folders inside their respective experiment directories as shown in the file structure. For example, the contents of `Global_025_CMEMS` from the download link should go into your local `Global_025_CMEMS` directory. The structure should look like this:
```
exp6_high_fidelity_eddy_forecast/
├── Global_025_CMEMS/
│ ├── model/ <-- Pre-trained weights here
│ ├── Inference_pdearena.py
│ └── ...
├── Regional_kuroshio_forcasting_0.125degree_daily/
│ ├── model/ <-- Pre-trained weights here
│ ├── inference_rollout.py
│ └── 
```
*Note: You may need to update the data and model paths inside the inference scripts if your directory structure is different.*

## ⚙️ Running Inference

### Part 1: Global 0.25° Forecasts

These scripts are located in the `Global_025_CMEMS/` directory.

- **Run TritonCast standard forecast:**
  ```bash
  python Global_025_CMEMS/Inference_tritoncast.py
  ```
- **Run baseline model (PDE-Refiner) forecast:**
  ```bash
  python Global_025_CMEMS/Inference_pdearena.py
  ```

- **Run the Zero-Shot Cross-Resolution Generalization Test:**
This test uses the model trained on 0.25° data to generate forecasts on unseen 0.125° high-resolution data.
  ```bash
  python Global_025_CMEMS/Inference_zero_shot_resolution.py
  ```

### Part 2: Regional High-Resolution 0.125° Forecasts
These scripts perform long rollouts for specific, dynamically complex ocean regions.

- **Kuroshio Current Forecast:**
  ```bash
  python Regional_kuroshio_forcasting_0.125degree_daily/inference_rollout.py
  ```

- **Gulf Stream Forecast:**
  ```bash
  python Regional_gulf_forcasting_0.125degree_daily/inference_rollout.py
  ```

- **Agulhas Current Forecast:**
  ```bash
  python Regional_agulhas_forcasting_0.125degree_daily/inference_rollout.py
  ```
  

