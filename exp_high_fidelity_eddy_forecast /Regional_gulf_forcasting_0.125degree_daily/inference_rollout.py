# -*- coding: utf-8 -*-
"""
Triton Model Multi-Initial Condition Rolling Inference Script

Functionality:
1. Load ocean current field data from a specified year.
2. Perform long-term rolling predictions (autoregressive) using a pre-trained Triton model.
3. Support setting multiple different initial conditions for multiple independent prediction experiments to evaluate model stability.
4. For each experiment, integrate and save the initial conditions, prediction results, and ground truth labels into a single NetCDF (.nc) file for subsequent analysis and visualization.
"""

import os
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
import logging
import math

# ==============================================================================
# --- 1. Global Configuration ---
# ==============================================================================

# --- Model & Path Configuration ---
BACKBONE = 'regional_gulf_forcasting_025degree_daily_20250808'
BASE_LOG_DIR = '/jizhicfs/Prometheus/Triton4Earth_V2/Scenario_3_Ocean_stream/regional_gulf_forcasting_0.125degree_daily'
CHECKPOINT_PATH = f'{BASE_LOG_DIR}/checkpoints/{BACKBONE}_best_model.pth'

# --- Output Directory Configuration ---
# Directory to store the final generated NetCDF result files
ROLLOUT_RESULTS_PATH = f'{BASE_LOG_DIR}/rollout_results_from_self'
os.makedirs(ROLLOUT_RESULTS_PATH, exist_ok=True)

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data & Model Hyperparameter Configuration ---
config = {
    'data_dir': '/jizhicfs/Prometheus/Triton4Earth_V2/Scenario_3_Ocean_stream/dataset/gulf_dataset',
    'input_steps': 10,          # Number of input days the model receives at once
    'output_steps': 10,         # Number of future days the model predicts at once
    'nan_fill_method': 'zero',  # NaN value filling method
    'downsample_factor': 2,     # Spatial downsampling factor (e.g., 2 means resolution halved)
    'forecast_year': 2023,      # Target year for forecasting
}

# --- Multi-Initial Condition Inference Configuration ---
# How many different initial condition experiments you want to run
NUM_INITIAL_CONDITIONS = 30
# Time step (days shifted back) between each initial condition
SHIFT_DAYS = 2

# --- Compute Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# --- 2. Model Definition ---
# ==============================================================================



from model.Triton_model import Triton


# ==============================================================================
# --- 3. Helper Functions ---
# ==============================================================================

def preprocess_data(ds: xr.Dataset, downsample_factor: int, nan_fill_method: str) -> np.ndarray:
    """
    Data preprocessing function, converts xarray Dataset to the numpy array required by the model.
    """
    # Extract ugos (meridional velocity) and vgos (zonal velocity) and fill NaN values
    ugos = np.nan_to_num(ds['ugos'].values, nan=0.0)
    vgos = np.nan_to_num(ds['vgos'].values, nan=0.0)
    
    # Stack the two physical quantities into one array, forming the format (time, channels, latitude, longitude)
    data = np.stack([ugos, vgos], axis=1)
    
    # If downsampling is set, perform downsampling on the latitude and longitude dimensions
    if downsample_factor > 1:
        data = data[:, :, ::downsample_factor, ::downsample_factor]
        
    return data

# ==============================================================================
# --- 4. Main Execution Function ---
# ==============================================================================

def main():
    """
    Main inference function, executes the complete process of model loading, data processing, loop prediction, and result saving.
    """
    logging.info(f"Starting multi-initial condition inference, total {NUM_INITIAL_CONDITIONS} runs.")
    logging.info(f"Using compute device: {DEVICE}")

    # --- Step 1: Load Model (Only need to load once) ---
    logging.info(f"Loading model from path: {CHECKPOINT_PATH}")
    h, w = 128, 128 # Downsampled image height and width, please confirm based on your data and downsampling factor
    model = Triton(
        shape_in=(config['input_steps'], 2, h, w),
        spatial_hidden_dim=256,
        output_channels=2,
        temporal_hidden_dim=512,
        num_spatial_layers=4,
        num_temporal_layers=8
    )
    if not os.path.exists(CHECKPOINT_PATH):
        logging.error(f"Model weights file not found: {CHECKPOINT_PATH}")
        return
        
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval() # Switch to evaluation mode, crucial for inference
    logging.info("Model loaded successfully.")

    # --- Step 2: Load Full Year Data Outside the Loop (Only need to load once) ---
    logging.info(f"Loading full year data for {config['forecast_year']}...")
    file_curr_path = os.path.join(config['data_dir'], f'ocean_currents_{config["forecast_year"]}.nc')
    if not os.path.exists(file_curr_path):
        logging.error(f"Data file not found: {file_curr_path}")
        return
    
    with xr.open_dataset(file_curr_path) as full_year_ds:
        total_days_in_year = len(full_year_ds.time)

        # --- Step 3: Create Main Loop, Process Each Initial Condition ---
        for i in range(NUM_INITIAL_CONDITIONS):
            # Calculate the starting day index for the current initial condition (0-based index)
            start_day_index = i * SHIFT_DAYS
            
            logging.info(f"\n===== Running {i+1}/{NUM_INITIAL_CONDITIONS} inference run (starting from day {start_day_index + 1}) =====")

            # Check if there is enough data to form a complete input
            if start_day_index + config['input_steps'] > total_days_in_year:
                logging.warning(f"Insufficient data to create initial condition starting from day {start_day_index + 1}. Stopping subsequent inference.")
                break

            # 3.1 Prepare data for the current loop
            # Slice the current initial condition and corresponding complete ground truth data from the full year data
            initial_data_raw = full_year_ds.isel(time=slice(start_day_index, start_day_index + config['input_steps']))
            ground_truth_raw = full_year_ds.isel(time=slice(start_day_index, None)) # All data from the current start point to the end of the year

            # Preprocess
            initial_input_np = preprocess_data(initial_data_raw, config['downsample_factor'], config['nan_fill_method'])
            full_ground_truth_np = preprocess_data(ground_truth_raw, config['downsample_factor'], config['nan_fill_method'])

            # 3.2 Perform step-by-step rolling prediction
            all_predictions = []
            current_input_tensor = torch.from_numpy(initial_input_np).float().to(DEVICE).unsqueeze(0)

            # Recalculate the number of days to predict and iterations needed for this prediction
            num_days_to_predict = len(full_ground_truth_np) - config['input_steps']
            if num_days_to_predict <= 0:
                logging.warning(f"Starting from day {start_day_index + 1}, there are no remaining days to predict. Skipping.")
                continue
            
            num_iterations = math.ceil(num_days_to_predict / config['output_steps'])
            
            with torch.no_grad(): # Turn off gradient calculation for speed and memory saving
                for _ in tqdm(range(num_iterations), desc=f"Rolling prediction (starting from day {start_day_index + 1})"):
                    predicted_chunk = model(current_input_tensor)
                    all_predictions.append(predicted_chunk.cpu().numpy())
                    current_input_tensor = predicted_chunk # Use output as next input

            # --- Step 4: Integrate and Save Current Loop Results to a Single .nc File ---
            logging.info("Integrating results and saving to NetCDF file...")

            # 4.1 Concatenate and crop prediction results
            final_predictions_np = np.concatenate(all_predictions, axis=1).squeeze(0)
            final_predictions_np = final_predictions_np[:num_days_to_predict]
            
            # 4.2 Prepare coordinate information
            total_time_steps_in_run = len(ground_truth_raw.time)
            downsampled_lat = ground_truth_raw.latitude.values[::config['downsample_factor']]
            downsampled_lon = ground_truth_raw.longitude.values[::config['downsample_factor']]

            # 4.3 Create empty numpy arrays to store the integrated data
            nan_fill = np.full((total_time_steps_in_run, h, w), np.nan, dtype=np.float32)
            initial_ugos, initial_vgos = nan_fill.copy(), nan_fill.copy()
            predicted_ugos, predicted_vgos = nan_fill.copy(), nan_fill.copy()
            
            # 4.4 Fill data
            initial_ugos[:config['input_steps']] = initial_input_np[:, 0, :, :]
            initial_vgos[:config['input_steps']] = initial_input_np[:, 1, :, :]
            predicted_ugos[config['input_steps']:] = final_predictions_np[:, 0, :, :]
            predicted_vgos[config['input_steps']:] = final_predictions_np[:, 1, :, :]
            ground_truth_ugos = full_ground_truth_np[:, 0, :, :]
            ground_truth_vgos = full_ground_truth_np[:, 1, :, :]

            # 4.5 Create xarray Dataset
            combined_ds = xr.Dataset(
                data_vars={
                    "initial_ugos": (("time", "latitude", "longitude"), initial_ugos, {"long_name": "Initial condition for u-component of velocity"}),
                    "initial_vgos": (("time", "latitude", "longitude"), initial_vgos, {"long_name": "Initial condition for v-component of velocity"}),
                    "predicted_ugos": (("time", "latitude", "longitude"), predicted_ugos, {"long_name": "Predicted u-component of velocity"}),
                    "predicted_vgos": (("time", "latitude", "longitude"), predicted_vgos, {"long_name": "Predicted v-component of velocity"}),
                    "ground_truth_ugos": (("time", "latitude", "longitude"), ground_truth_ugos, {"long_name": "Ground truth u-component of velocity"}),
                    "ground_truth_vgos": (("time", "latitude", "longitude"), ground_truth_vgos, {"long_name": "Ground truth v-component of velocity"}),
                },
                coords={
                    "time": ground_truth_raw.time.values,
                    "latitude": downsampled_lat,
                    "longitude": downsampled_lon,
                }
            )

            # 4.6 Save to .nc file
            file_suffix = f'start_day_{start_day_index+1}'
            output_nc_path = os.path.join(ROLLOUT_RESULTS_PATH, f'{BACKBONE}_{file_suffix}_full_results.nc')
            combined_ds.to_netcdf(output_nc_path)
            
            logging.info(f"Full results for start day {start_day_index+1} saved to: {output_nc_path}")

    logging.info("\nAll inference tasks completed successfully.")

# ==============================================================================
# --- 5. Script Entry Point ---
# ==============================================================================

if __name__ == '__main__':
    main()