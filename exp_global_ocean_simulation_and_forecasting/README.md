

 
 # <p align=center> Inference code for global ocean simulation and forecasting</p>

 

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
|--networks
|--my_utils
|--exp
|--inference_simulation.sh
|--inference_simulation.py
|--inference_forecasting.sh
|--inference_forecasting.py
|--environment.yml
```

2. Download the whole `exp` file from this [link](https://huggingface.co/TritonCast/TritonCast_model/tree/main/exp_global_ocean_simulation_and_forecasting/exp).

3. Download the test data, pre-trained checkpoints from this [link1](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Ocean%20Simulation%20and%20Forecasting) and [link2](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Medium-range%20Weather%20Forecasting), and put them into a folder. (Note: `2020.h5` in link1 and `climate_mean_s_t_ssh.npy` are multi-part compressed files; you need to decompress them first.)

4. Modify the following line in `my_utils.data_loader_multifiles.py` using your own path.

    self.climate_mean = np.load('/apdcephfs_qy3/share_301734960/easyluwu/gy/data/coupled_0.25_23layers/climate_mean_s_t_ssh.npy', mmap_mode='r')

5. Modify the following line in `exp/Triton/20250819-012814/config.yaml` using your own path.

    exp_dir: '/apdcephfs_qy3/share_301734960/easyluwu/gy/triton_v2/exp_global_ocean_simulation_and_forecasting/exp'

    test_data_path:  '/apdcephfs_qy3/share_301734960/easyluwu/gy/data/coupled_0.25_23layers/test' (`2020.h5` in this [link1](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Ocean%20Simulation%20and%20Forecasting))

    test_data_path_atmos:  '/jizhicfs/easyluwu/scaling_law/ft_local/weatherbench2/121_240/69var/test'  (`2020.h5` in this [link2](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Medium-range%20Weather%20Forecasting))

    land_mask_path: '/apdcephfs_qy3/share_301734960/easyluwu/gy/data/coupled_0.25_23layers/land_mask.h5'

    global_means_path: '/apdcephfs_qy3/share_301734960/easyluwu/gy/data/coupled_0.25_23layers/mean_s_t_ssh.npy' 

    global_stds_path:  '/apdcephfs_qy3/share_301734960/easyluwu/gy/data/coupled_0.25_23layers/std_s_t_ssh.npy'

    global_means_path_atmos: '/jizhicfs/easyluwu/scaling_law/ft_local/weatherbench2/121_240/69var/mean.npy' 
    
    global_stds_path_atmos:  '/jizhicfs/easyluwu/scaling_law/ft_local/weatherbench2/121_240/69var/std.npy'


### Inference for Global Ocean Simulation

Run the following script:

```
sh inference_simulation.sh
```

### Inference for Global Ocean Forecasting

Run the following script:

```
sh inference_forecasting.sh
```

   
