Here's the modified code with English translations for the comments while keeping all other content unchanged:

```python
import numpy as np
import netCDF4 as nc
import xarray as xr
from tqdm import tqdm

def normalize_and_save_data(years, params=None, norm_prefix="_norm", mode="train"):
    """
    Normalize and save data, initial data is saved in files like {year}.nc
    
    Parameters:
    years: year range
    params: normalization parameters array
    norm_prefix: suffix for normalized files
    mode: running mode ('train', 'val', 'test')
    """
    for year in tqdm(years, desc=f"Processing {mode} data"):
        with nc.Dataset(f"{year}.nc") as ds:
            var_data = ds.variables['atmosphere_variables'][:]
            
            # Get normalization parameter index
            idx = year - 1979
            
            for var_idx in range(var_data.shape[1]):
                if mode == "train":
                    # Training set: calculate and store mean and std
                    mean_val = np.mean(var_data[:, var_idx])
                    std_val = np.std(var_data[:, var_idx])
                    params[idx, var_idx, 0] = mean_val
                    params[idx, var_idx, 1] = std_val
                else:
                    # Validation/test set: use pre-computed parameters
                    mean_val = params[idx, var_idx, 0]
                    std_val = params[idx, var_idx, 1]
                
                # Normalize data
                var_data[:, var_idx] = (var_data[:, var_idx] - mean_val) / std_val
            
            # Create and save xarray dataset
            ds_out = xr.Dataset({
                'atmosphere_variables': (['time', 'var', 'latitude', 'longitude'], var_data)
            })
            ds_out.to_netcdf(f"{year}{norm_prefix}.nc")

# Main processing flow
if __name__ == "__main__":
    # Initialize normalization parameters array
    normalized_params = np.zeros((43, 69, 2))  # [year, variable, (mean, std)]
    
    # Process training data (1979-2017)
    train_years = range(1979, 2018)
    normalize_and_save_data(train_years, normalized_params, mode="train")
    
    # Calculate normalization parameters for validation/test sets (using 1979-2017 average)
    avg_params = np.mean(normalized_params[:len(train_years)], axis=0)
    normalized_params[len(train_years):] = avg_params
    
    # Process validation and test data (2018-2021)
    test_years = range(2018, 2022)
    normalize_and_save_data(test_years, normalized_params, mode="test")
    
    # Optional: save normalization parameters
    np.save("params.npy", normalized_params)
```