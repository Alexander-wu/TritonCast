# Year-long Weather Forecasting Model Deployment Technical Implementation Document

## Quick Start


The code structure is as follows:
```
exp_long_term_stability_test/
├── checkpoints/
├── dataloader_api/
├── logs/
├── model/
├── model_baselines/
├── README.md
├── data_norm.py
├── inference.py
├── params.npy
```


Follow these steps to run the code:

1.  **Download Pre-trained Weights**: [Click here](https://huggingface.co/TritonCast/TritonCast_model/tree/main/exp_long_term_stability_test) to download our provided pre-trained weights and place them in the `checkpoints` directory. Alternatively, you can train your own model by following our tutorial.
2.  **Download the Inference Dataset**: [Click here](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Atmospheric_year_long_forecast) to download the dataset required for inference, create a new folder named `dataset`, and place it in the same root directory as the `checkpoints` folder.
3.  **Modify Configuration**: Open the `inference.py` file and update the paths to the weights and the dataset to match your local environment.

**Important Recommendation**:
For optimal results, we highly recommend running the inference on hardware that supports high-precision computing, such as NVIDIA A800, A100, or H20 GPUs. The annual cycle prediction task is extremely sensitive to numerical precision, and even a minor loss in accuracy can significantly degrade performance. Our tests on the aforementioned hardware have successfully reproduced the results reported in the paper.

## Data Preprocessing Specifications

### 1. Input Feature Order

The channel order of model input data is as follows (total 69 channels):
0: u10 (10m U-component of wind)
1: v10 (10m V-component of wind)
2: T2m (2m temperature)
3: msl (Mean sea level pressure)
4-16: 13 pressure levels of U-component
17-29: 13 pressure levels of V-component
30-42: 13 pressure levels of temperature
43-55: 13 pressure levels of geopotential
56-68: 13 pressure levels of specific humidity


### 2. Pressure Levels Arrangement

- Pressure levels are arranged **from low to high**
- Consistent with Pangu model configuration

### 3. Data Normalization

- Using **z-score normalization** method
- Normalization parameters stored in [params.npy](params.npy) file
- Data processing script: [data_norm.py](data_norm.py)

## Data Source Processing Specifications

### 1. ERA5 Data Processing

- Using **daily instantaneous data at UTC 12:00**
- **No daily averaging** performed
- Reference processing script: [data_norm.py](data_norm.py)

### 2. Other Data Sources (e.g. EC) Adaptation

- Recommended to modify based on ERA5 processing script
- Key implementation requirements:
  - Spatiotemporal alignment
  - Channel order matching
  - Normalization parameters adaptation

## Implementation Recommendations

1. Obtain historical data statistics [params.npy](params.npy) for z-score normalization
2. Perform data preprocessing using provided [data_norm.py](data_norm.py) script
3. For non-ERA5 data sources, focus on:
   - Temporal resolution alignment (UTC 12:00 instantaneous values)
   - Spatial resolution matching
   - Variable channel order adjustment

## Important Notes

- Pressure level data must be arranged from low to high
- Daily forecasts use instantaneous values rather than daily averages
- Ensure spatiotemporal alignment and variable consistency when converting between different data sources
