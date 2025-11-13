import os
import torch
import numpy as np
import xarray as xr
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd 
from models.Triton_ocean import Triton_Ocean_lighted

config = {
    'data_root': '/apdcephfs_qy3/share_301734960/easyluwu/shuruiqi/NeuralPS_25/CMEMS/low',
    'model_path': '/jizhicfs/easyluwu/Triton4Earth_V2/Scenario_3_Ocean_stream/Global_025_stream/checkpoints/light_Triton_global_ocean_20250819_uv_0.25_best_model.pth',
    'output_path': '/jizhicfs/easyluwu/Triton4Earth_V2/Scenario_3_Ocean_stream/Global_025_stream/inference_results',
    'device': 'cpu' if torch.cuda.is_available() else 'cpu', 
    'input_steps': 10,
    'pred_steps': 10, 
    'variables': ['ugos', 'vgos'],
    'target_vars': ['ugos', 'vgos'],
    'lon_range': (0, 1440),
    'lat_range': (0, 720),
    'downsample': 1,
    'inference_year': 2023,
    'num_autoregressive_steps': 3,
    'num_initial_conditions': 5,
    'initial_condition_stride': 1,
}
total_pred_days = config['pred_steps'] * config['num_autoregressive_steps']

try:
    print(f"Loading model from: {config['model_path']}")
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    model = Triton_Ocean_lighted(
        input_time=config['input_steps'], 
        spatial_hidden_dim=64, 
        output_channels=len(config['target_vars']),
        temporal_hidden_dim=256,
        num_spatial_layers=4, 
        num_temporal_layers=4
    )
    checkpoint = torch.load(config['model_path'], map_location=device)
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

all_initial_conditions_list = []
all_predictions_list = []
all_ground_truths_list = []
all_initial_times_list = []
all_prediction_times_list = []

inference_data_path = os.path.join(config['data_root'], f"SSH_low_processed_{config['inference_year']}.nc")
print(f"Loading data for the entire year {config['inference_year']} from: {inference_data_path}")

inference_year = config['inference_year']
num_days_in_year = 366 if pd.to_datetime(f'{inference_year}-01-01').is_leap_year else 365
full_year_dates = pd.date_range(start=f'{inference_year}-01-01', periods=num_days_in_year, freq='D')
lat_coords_real = np.linspace(89.875, -89.875, 720)
lon_coords_real = np.linspace(0.125, 359.875, 1440)

try:
    with xr.open_dataset(inference_data_path) as ds:
        total_days_in_year_from_file = len(ds['time'])
        max_start_day = total_days_in_year_from_file - (config['input_steps'] + total_pred_days)
        start_days = range(0, max_start_day + 1, config['initial_condition_stride'])
        
        if len(start_days) > config['num_initial_conditions']:
            start_days = start_days[:config['num_initial_conditions']]
        
        num_valid_starts = len(start_days)
        print(f"Found {num_valid_starts} valid initial conditions to test.")
        if num_valid_starts == 0:
            print("Warning: No valid initial conditions found. The program will not perform inference and save.")
            print(f"Please check: Is the total number of days in the data file ({total_days_in_year_from_file}) less than the total required days ({config['input_steps'] + total_pred_days})")


        for start_day in tqdm(start_days, desc="Processing Initial Conditions"):
            end_input_day = start_day + config['input_steps']
            
            initial_input_data = ds[config['variables']].isel(
                time=slice(start_day, end_input_day),
                lon=slice(*config['lon_range'], config['downsample']),
                lat=slice(*config['lat_range'], config['downsample'])
            ).to_array().values
            
            initial_input_tensor = torch.tensor(initial_input_data, dtype=torch.float32).permute(1, 0, 2, 3)
            initial_input_tensor = torch.nan_to_num(initial_input_tensor, nan=0.0)
            all_initial_conditions_list.append(initial_input_tensor.unsqueeze(0))
            
            initial_times = full_year_dates[start_day:end_input_day].to_numpy()
            all_initial_times_list.append(initial_times)
            
            current_input = initial_input_tensor.unsqueeze(0).to(device)
            single_case_predictions = []
            with torch.no_grad():
                for _ in range(config['num_autoregressive_steps']):
                    predicted_output = model(current_input)
                    single_case_predictions.append(predicted_output.cpu())
                    current_input = predicted_output.to(device)
            prediction_tensor = torch.cat(single_case_predictions, dim=1)
            all_predictions_list.append(prediction_tensor)

            start_gt_day = end_input_day
            end_gt_day = start_gt_day + total_pred_days
            gt_data = ds[config['target_vars']].isel(
                time=slice(start_gt_day, end_gt_day),
                lon=slice(*config['lon_range'], config['downsample']),
                lat=slice(*config['lat_range'], config['downsample'])
            ).to_array().values
            gt_tensor = torch.tensor(gt_data, dtype=torch.float32).permute(1, 0, 2, 3)
            gt_tensor = torch.nan_to_num(gt_tensor, nan=0.0)
            all_ground_truths_list.append(gt_tensor.unsqueeze(0))
            
            prediction_times = full_year_dates[start_gt_day:end_gt_day].to_numpy()
            all_prediction_times_list.append(prediction_times)
except FileNotFoundError:
    print(f"Error: Data file not found at {inference_data_path}. Please check the path.")
    exit() # Exit directly if the data file is not found

# ========================== 4. Aggregate data and save (with error handling) ==========================
num_initial_conditions = 30
if all_predictions_list:
    num_cases = len(all_predictions_list)
    initial_conditions_arr = torch.cat(all_initial_conditions_list, dim=0).numpy()
    predictions_arr = torch.cat(all_predictions_list, dim=0).numpy()
    ground_truth_arr = torch.cat(all_ground_truths_list, dim=0).numpy()
    initial_times_arr = np.stack(all_initial_times_list, axis=0)
    prediction_times_arr = np.stack(all_prediction_times_list, axis=0)

    output_ds = xr.Dataset(
        data_vars={
            'initial_condition': (('case', 'input_time_step', 'variable', 'lat', 'lon'), initial_conditions_arr),
            'prediction': (('case', 'lead_time_step', 'variable', 'lat', 'lon'), predictions_arr),
            'ground_truth': (('case', 'lead_time_step', 'variable', 'lat', 'lon'), ground_truth_arr),
        },
        coords={
            'case': np.arange(num_cases),
            'input_time': (('case', 'input_time_step'), initial_times_arr),
            'time': (('case', 'lead_time_step'), prediction_times_arr),
            'lead_time': ('lead_time_step', np.arange(1, total_pred_days + 1)),
            'variable': config['target_vars'],
            'lat': lat_coords_real,  
            'lon': lon_coords_real,  
        }
    )
    
    output_ds.attrs['description'] = f"Autoregressive forecast results from model {config['model_path']}"
    output_ds.attrs['inference_year'] = config['inference_year']
    output_ds['lead_time'].attrs['units'] = 'days'
    output_ds['time'].attrs['long_name'] = 'valid time for prediction and ground truth'
    output_ds['input_time'].attrs['long_name'] = 'valid time for initial condition'

    try:
        os.makedirs(config['output_path'], exist_ok=True)
        save_path = os.path.join(config['output_path'], f'tritoncast_inference_results_{num_initial_conditions}_2023.nc') 
        print(f"\nAttempting to save the results to: {save_path}")
        output_ds.to_netcdf(save_path)
        print("Results saved successfully!")
    except Exception as e:
        print("\n!!!!!!!!!! An error occurred while saving the file !!!!!!!!!!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print("Please check:")
        print(f"1. Does the path '{config['output_path']}' exist and do you have write permissions?")
        print("2. Is there enough disk space?")

else:
    print("\nNo valid initial conditions were found. Nothing to save.")

print("\nAll tasks finished.")