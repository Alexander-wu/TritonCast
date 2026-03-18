import os
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
from models.PDEArena import *

# ========================== Inference Configuration ==========================
model_type = "light_04B"
REGION_NAME = f'{model_type}_pdearena_global_ocean'
backbone = f'{REGION_NAME}_uv_0.25'

config = {
    'data_root': '/apdcephfs_qy3/share_301734960/easyluwu/shuruiqi/NeuralPS_25/CMEMS/low',
    'model_path': f'/jizhicfs/easyluwu/Triton4Earth_V2/Scenario_3_Ocean_stream/Global_025_stream/checkpoints/{backbone}_best_model.pth',
    'output_path': '/jizhicfs/easyluwu/Triton4Earth_V2/Scenario_3_Ocean_stream/Global_025_stream/inference_results',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'input_steps': 10,
    'pred_steps': 10,
    'variables': ['ugos', 'vgos'],
    'target_vars': ['ugos', 'vgos'],
    'lon_range': (0, 1440),
    'lat_range': (0, 720),
    'downsample': 1,
    'inference_year': 2023,  # Year for inference
    'num_autoregressive_steps': 3,  # Number of autoregressive steps
    'num_initial_conditions': 30,  # Number of initial conditions
    'initial_condition_stride': 2,  # Stride for initial conditions
}

# Calculate total prediction days
total_pred_days = config['pred_steps'] * config['num_autoregressive_steps']

# ========================== Load Model ==========================
try:
    print(f"Loading model from: {config['model_path']}")
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    
    # Use the same model configuration as in training
    common_args = {
        "n_input_scalar_components": 0,
        "n_input_vector_components": 1,
        "n_output_scalar_components": 0,
        "n_output_vector_components": 1,
        "time_history": config['input_steps'],
        "time_future": config['pred_steps'],
        "activation": "silu",
    }
    
    fourier_unet_init_args = {
        "hidden_channels": 32,
        "modes1": 8,
        "modes2": 8,
        "norm": True,
        "n_fourier_layers": 1,
    }
    
    all_config_args = {**common_args, **fourier_unet_init_args}
    model = FourierUnet(**all_config_args)
    
    # Load model weights
    checkpoint = torch.load(config['model_path'], map_location=device)
    
    # Handle weights saved from distributed training (remove 'module.' prefix)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
except FileNotFoundError:
    print(f"Error: Model file not found at {config['model_path']}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    exit()

# ========================== Prepare Data Containers ==========================
all_initial_conditions_list = []
all_predictions_list = []
all_ground_truths_list = []
all_initial_times_list = []
all_prediction_times_list = []

# Latitude and longitude coordinates (adjust according to actual data)
lat_coords_real = np.linspace(89.875, -89.875, 720)
lon_coords_real = np.linspace(0.125, 359.875, 1440)

# ========================== Load Data and Perform Inference ==========================
inference_data_path = os.path.join(config['data_root'], f"SSH_low_processed_{config['inference_year']}.nc")
print(f"Loading data for the entire year {config['inference_year']} from: {inference_data_path}")

try:
    # Get information about the number of days in the year
    inference_year = config['inference_year']
    num_days_in_year = 366 if pd.to_datetime(f'{inference_year}-01-01').is_leap_year else 365
    full_year_dates = pd.date_range(start=f'{inference_year}-01-01', periods=num_days_in_year, freq='D')
    
    with xr.open_dataset(inference_data_path) as ds:
        total_days_in_year_from_file = len(ds['time'])
        max_start_day = total_days_in_year_from_file - (config['input_steps'] + total_pred_days)
        
        # Generate starting points for initial conditions
        start_days = range(0, max_start_day + 1, config['initial_condition_stride'])
        
        # Limit the number of initial conditions
        if len(start_days) > config['num_initial_conditions']:
            start_days = start_days[:config['num_initial_conditions']]
        
        num_valid_starts = len(start_days)
        print(f"Found {num_valid_starts} valid initial conditions to test.")
        
        if num_valid_starts == 0:
            print("Warning: No valid initial conditions found. The program will not perform inference and save.")
            print(f"Please check: Is the total number of days in the data file ({total_days_in_year_from_file}) less than the total required days ({config['input_steps'] + total_pred_days})")
            exit()
        
        # Perform inference for each initial condition
        for start_day in tqdm(start_days, desc="Processing Initial Conditions"):
            end_input_day = start_day + config['input_steps']
            
            # Load initial input data
            initial_input_data = ds[config['variables']].isel(
                time=slice(start_day, end_input_day),
                lon=slice(*config['lon_range'], config['downsample']),
                lat=slice(*config['lat_range'], config['downsample'])
            ).to_array().values
            
            # Preprocess data (consistent with training)
            initial_input_tensor = torch.tensor(initial_input_data, dtype=torch.float32)
            initial_input_tensor = initial_input_tensor.permute(1, 0, 2, 3)  # (T_in, C, H, W)
            initial_input_tensor = torch.nan_to_num(initial_input_tensor, nan=0.0)
            
            all_initial_conditions_list.append(initial_input_tensor.unsqueeze(0))  # Add batch dimension
            
            # Record initial time
            initial_times = full_year_dates[start_day:end_input_day].to_numpy()
            all_initial_times_list.append(initial_times)
            
            # Autoregressive prediction
            current_input = initial_input_tensor.unsqueeze(0).to(device)  # (1, T_in, C, H, W)
            single_case_predictions = []
            
            with torch.no_grad():
                for _ in range(config['num_autoregressive_steps']):
                    # Model prediction
                    predicted_output = model(current_input)
                    single_case_predictions.append(predicted_output.cpu())
                    
                    # Update input: use the latest prediction as input for the next time step
                    # Assuming the model output shape is (batch, T_pred, C, H, W)
                    current_input = torch.cat([current_input[:, config['pred_steps']:, :, :, :], 
                                             predicted_output.to(device)], dim=1)
            
            # Concatenate all prediction results
            prediction_tensor = torch.cat(single_case_predictions, dim=1)
            all_predictions_list.append(prediction_tensor)
            
            # Load ground truth for validation
            start_gt_day = end_input_day
            end_gt_day = start_gt_day + total_pred_days
            gt_data = ds[config['target_vars']].isel(
                time=slice(start_gt_day, end_gt_day),
                lon=slice(*config['lon_range'], config['downsample']),
                lat=slice(*config['lat_range'], config['downsample'])
            ).to_array().values
            
            gt_tensor = torch.tensor(gt_data, dtype=torch.float32)
            gt_tensor = gt_tensor.permute(1, 0, 2, 3)  # (T_pred, C, H, W)
            gt_tensor = torch.nan_to_num(gt_tensor, nan=0.0)
            all_ground_truths_list.append(gt_tensor.unsqueeze(0))  # Add batch dimension
            
            # Record prediction time
            prediction_times = full_year_dates[start_gt_day:end_gt_day].to_numpy()
            all_prediction_times_list.append(prediction_times)
            
except FileNotFoundError:
    print(f"Error: Data file not found at {inference_data_path}. Please check the path.")
    exit()
except Exception as e:
    print(f"An error occurred during data processing or inference: {e}")
    exit()

# ========================== Save Results ==========================
if all_predictions_list:
    # Aggregate all results
    initial_conditions_arr = torch.cat(all_initial_conditions_list, dim=0).numpy()
    predictions_arr = torch.cat(all_predictions_list, dim=0).numpy()
    ground_truth_arr = torch.cat(all_ground_truths_list, dim=0).numpy()
    initial_times_arr = np.stack(all_initial_times_list, axis=0)
    prediction_times_arr = np.stack(all_prediction_times_list, axis=0)
    
    num_cases = len(all_predictions_list)
    
    # Create xarray dataset
    output_ds = xr.Dataset(
        data_vars={
            'initial_condition': (('case', 'input_time_step', 'variable', 'lat', 'lon'), initial_conditions_arr),
            'prediction': (('case', 'lead_time_step', 'variable', 'lat', 'lon'), predictions_arr),
            'ground_truth': (('case', 'lead_time_step', 'variable', 'lat', 'lon'), ground_truth_arr),
        },
        coords={
            'case': np.arange(num_cases),
            'input_time_step': np.arange(config['input_steps']),
            'lead_time_step': np.arange(total_pred_days),
            'input_time': (('case', 'input_time_step'), initial_times_arr),
            'time': (('case', 'lead_time_step'), prediction_times_arr),
            'variable': config['target_vars'],
            'lat': ('lat', lat_coords_real[config['lat_range'][0]:config['lat_range'][1]:config['downsample']]),
            'lon': ('lon', lon_coords_real[config['lon_range'][0]:config['lon_range'][1]:config['downsample']]),
        }
    )
    
    # Add attribute information
    output_ds.attrs['description'] = f"Autoregressive forecast results from FourierUnet model {config['model_path']}"
    output_ds.attrs['inference_year'] = config['inference_year']
    output_ds.attrs['model_type'] = model_type
    output_ds.attrs['backbone'] = backbone
    
    # Save results
    try:
        os.makedirs(config['output_path'], exist_ok=True)
        save_path = os.path.join(config['output_path'], f'{backbone}_inference_results_{config["inference_year"]}.nc')
        print(f"\nSaving results to: {save_path}")
        output_ds.to_netcdf(save_path)
        print("Results saved successfully!")
        
    except Exception as e:
        print(f"\nError saving file: {e}")
        print("Please check:")
        print(f"1. Path '{config['output_path']}' exists and is writable")
        print(f"2. Disk space is sufficient")

else:
    print("\nNo valid initial conditions were found. Nothing to save.")

print("\nInference completed!")