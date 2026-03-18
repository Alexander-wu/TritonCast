

 
 # <p align=center> Inference code for medium-range weather forecasting</p>

 

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
|--models
|--exp
|--inference.sh
|--inference.py
|--environment.yml
```

2. Download the whole `exp` file from this [link](https://huggingface.co/TritonCast/TritonCast_model/tree/main/exp_medium_range_weather_forecasting).

3. Download the test data from this [link](https://huggingface.co/datasets/TritonCast/TritonCast_inference_datasets/tree/main/Medium-range%20Weather%20Forecasting) and put them into a folder.

5. Modify the following line in `exp/Triton/20250731-162648/config.yaml` using your own path.

    test_data_path:  '/jizhicfs/easyluwu/scaling_law/ft_local/weatherbench2/121_240/69var/test'

    global_means_path: '/jizhicfs/easyluwu/scaling_law/ft_local/weatherbench2/121_240/69var/mean.npy' 

    global_stds_path:  '/jizhicfs/easyluwu/scaling_law/ft_local/weatherbench2/121_240/69var/std.npy'
    


### Inference for Global Medium-range Weather Forecasting

Run the following script:

```
sh inference.sh
```

   
