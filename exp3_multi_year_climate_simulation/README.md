

 
 # <p align=center> Inference code for multi-year climate simulation</p>

 

## Quick Start

### Installation

- cuda 11.8

```
# create new anaconda env
conda env create -f environment.yml
conda activate triton_v2
```

### Preparing test data and pre-trained checkpoints

1. The code structure is as follows:


```
./
|--my_utils
|--networks
|--exp
|--inference.sh
|--inference.py
|--environment.yml
```

2. Download the whole `exp` file from this [link](https://huggingface.co/TritonCast/TritonCast_model/tree/main/exp_multi_year_climate_simulation).

3. Download the test data from this [link](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Multi-year%20Climate%20Simulation) and put them into a folder.

5. Modify the following line in `exp/Triton/20250803-152259/config.yaml` using your own path.

    test_data_path:  '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/weatherbench2/69var_1.5_degree_1day/test_2018_to_2024' (`2018_to_2024_atmos.h5`)

    test_data_path_ocean:  '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/coupled_1.5_23layers/test_2018_to_2024' (`2018_to_2024_ocean.h5`)

    land_mask_path: '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/coupled_1.5_23layers/land_mask.h5'

    global_means_path: '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/weatherbench2/69var_1.5_degree_1day/mean_atmos.npy' 

    global_stds_path:  '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/weatherbench2/69var_1.5_degree_1day/std_atmos.npy' 

    global_means_path_ocean: '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/coupled_1.5_23layers/mean_ocean.npy' 

    global_stds_path_ocean:  '/jizhicfs/Prometheus/gaoyuan/llm/ft_local/data/coupled_1.5_23layers/std_ocean.npy'
    


### Inference for Global Medium-range Weather Forecasting

Run the following script:

```
sh inference.sh
```

   
