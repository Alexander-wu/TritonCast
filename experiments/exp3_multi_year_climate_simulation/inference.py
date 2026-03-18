import os
import sys
import time
import glob
import h5py
# import wandb
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from datetime import datetime
from numpy.core.numeric import False_

import torch
import torchvision
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torchvision.utils import save_image
from torch.nn.parallel import DistributedDataParallel


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../../')
from my_utils.YParams import YParams
from my_utils.data_loader_multifiles import get_data_loader

from my_utils import logging_utils
from my_utils import time_utils 
from tritoncast.models import build_model
from tritoncast.utils import load_model_checkpoint

logging_utils.config_logger()

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    return load_model_checkpoint(model, checkpoint_file)


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

    # get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.test_data_path, params.test_data_path_ocean, dist.is_initialized(), train=False)

    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)


    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels

    if params.normalization == 'zscore': 
        params.means = np.load(params.global_means_path)
        params.stds = np.load(params.global_stds_path)
        
        params.means_ocean = np.load(params.global_means_path_ocean)
        params.stds_ocean = np.load(params.global_stds_path_ocean)
    
    if params.nettype != 'Triton':
        raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    logging.info('Loading trained model checkpoint from {}'.format(checkpoint_file))
    model = build_model("Triton", experiment="multi_year_climate", params=params).to(device)
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    files_paths = glob.glob(params.test_data_path + "/*.h5")
    files_paths_ocean = glob.glob(params.test_data_path_ocean + "/*.h5")
    files_paths.sort()
    files_paths_ocean.sort()

    # which year
    yr = 0
    logging.info('Loading inference data')
    logging.info('Inference data from {}'.format(files_paths[yr]))
    valid_data_full = h5py.File(files_paths[yr], 'r')['fields'][:, :, :, :]
    valid_data_full_ocean = h5py.File(files_paths_ocean[yr], 'r')['fields'][:, :, :, :]

    return valid_data_full, valid_data_full_ocean, model


    
def autoregressive_inference(params, init_condition, valid_data_full, valid_data_full_ocean, model): 
    icd = int(init_condition) 
    
    # initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt                = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history      = params.n_history
    img_shape_x    = params.img_shape_x
    img_shape_y    = params.img_shape_y
    in_channels    = np.array(params.in_channels)
    out_channels   = np.array(params.out_channels)
    atmos_channels = np.array(params.atmos_channels)
    n_in_channels  = len(in_channels)
    n_out_channels = len(out_channels)

    seq_real        = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))
    seq_pred        = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y))

    # extract valid data 
    valid_day = np.arange(0, 365)[icd:(icd+prediction_length*dt+n_history*dt):dt]
    logging.info(f'valid_day: {valid_day}')
    valid_date = [time_utils.get_date(params.year, day) for day in valid_day]
    logging.info(f'valid_date: {valid_date}')

    valid_data = valid_data_full[icd:(icd+prediction_length*dt+n_history*dt):dt][:, params.in_channels][:,:,0:120]
    valid_data_ocean = valid_data_full_ocean[icd:(icd+prediction_length*dt+n_history*dt):dt][:, :][:,:,0:120]
    logging.info(f'valid_data_full: {valid_data_full.shape}')
    logging.info(f'valid_data: {valid_data.shape}')
    
    logging.info(f'valid_data_full_ocean: {valid_data_full_ocean.shape}')
    logging.info(f'valid_data_ocean: {valid_data_ocean.shape}')
    
    # normalize
    if params.normalization == 'zscore': 
        valid_data = (valid_data - params.means[:,params.in_channels])/params.stds[:,params.in_channels]
        valid_data = np.nan_to_num(valid_data, nan=0)
        
        valid_data_ocean = (valid_data_ocean - params.means_ocean[:,:])/params.stds_ocean[:,:]
        valid_data_ocean = np.nan_to_num(valid_data_ocean, nan=0)
        
    valid_data = torch.as_tensor(valid_data)
    valid_data_ocean = torch.as_tensor(valid_data_ocean)

    # autoregressive inference
    logging.info('Begin autoregressive inference')
    
    
    with torch.no_grad():
        for i in range(prediction_length): 
            print(i)
            if i==0: # start of sequence, t0 --> t0'
                first = valid_data[0:n_history+1]
                first_ocean = valid_data_ocean[0:n_history+1]
                ic(valid_data.shape, first.shape)
                future = valid_data[n_history+1]
                ic(future.shape)

                for h in range(n_history+1):
                    
                    seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels, :93] # extract history from 1st
                    
                    seq_pred[h] = seq_real[h]

               
                
                first = first.to(device, dtype=torch.float)
                first_atmosphere = first[:, params.in_channels, :, :]
                ic(first_atmosphere.shape)
                first_ocean = first_ocean[:, params.ocean_channels, :, :].to(device, dtype=torch.float)
                ic(first_ocean.shape)
                # first_ocean = first_ocean[params.ocean_channels, :120, :240]
                # first_ocean = torch.unsqueeze(first_ocean, dim=0).to(device, dtype=torch.float)
                model_input = torch.cat((first_atmosphere, first_ocean.cuda()), axis=1)
                ic(model_input.shape)
                future_pred = model(model_input)
                print(future_pred.shape, first_ocean.shape, torch.mean(first_ocean))

            else: # (t1) --> (t+1)', (t+1)' --> (t+2)', (t+2)' --> (t+3)' ....
                if i < prediction_length-1:
                    future = valid_data[n_history+i+1]
                ocean = valid_data_ocean[n_history+i]

               
                inf_one_step_start = time.time()
                ocean = ocean[params.ocean_channels, :120, :240]
                ocean = torch.unsqueeze(ocean, dim=0).to(device, dtype=torch.float)
                print(future_pred.shape, ocean.shape, torch.mean(ocean))
                future_pred = model(torch.cat((future_pred.cuda(), ocean), axis=1)) #autoregressive step
                inf_one_step_time = time.time() - inf_one_step_start

                logging.info(f'inference one step time: {inf_one_step_time}')

            if i < prediction_length - 1: # not on the last step
                
                seq_pred[n_history+i+1] = future_pred
                
                seq_real[n_history+i+1] = future[:]
                history_stack = seq_pred[i+1:i+2+n_history]

            future_pred = history_stack

            pred = torch.unsqueeze(seq_pred[i], 0)
            tar  = torch.unsqueeze(seq_real[i], 0)

            print(torch.mean((pred-tar)**2))



    
    seq_real = seq_real * params.stds[:,params.out_channels] + params.means[:,params.out_channels]
    seq_real = seq_real.numpy()
    seq_pred = seq_pred * params.stds[:,params.out_channels] + params.means[:,params.out_channels]
    seq_pred = seq_pred.numpy()

    return (np.expand_dims(seq_real[n_history:], 0), 
            np.expand_dims(seq_pred[n_history:], 0) )  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", default='../exp_15_levels', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--run_num", default='00', type=str)
    # parser.add_argument("--yaml_config", default='../config/AFNO.yaml', type=str)
    parser.add_argument("--prediction_length", default=30, type=int)
    parser.add_argument("--decorrelation_time", default=30, type=int)
    parser.add_argument("--n_samples_per_year", default=365, type=int)
    parser.add_argument("--finetune_dir", default='', type=str)

    parser.add_argument("--ics_type", default='default', type=str)
    parser.add_argument("--year", default=2012, type=int)
    parser.add_argument("--date_strings", default='01/01/2021 00:00:00,01/02/2021 00:00:00,01/03/2021 00:00:00', type=str)
    args = parser.parse_args()

    config_path = os.path.join(args.exp_dir, args.config, args.run_num, 'config.yaml')
    params = YParams(config_path, args.config)

    params['resuming']           = False
    params['interp']             = 0 
    params['world_size']         = 1
    params['local_rank']         = 0
    params['global_batch_size']  = params.batch_size
    params['prediction_length']  = args.prediction_length
    params['decorrelation_time'] = args.decorrelation_time
    params['n_samples_per_year'] = args.n_samples_per_year
    params['multi_steps_finetune'] = 1
    params['year']         = args.year
    params['ics_type']     = args.ics_type
    params['date_strings'] = args.date_strings.split(",")

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.finetune_dir == '':
        expDir = os.path.join('/jizhicfs/Prometheus/gaoyuan/llm/ft_local/NeuralOA', params.exp_dir, args.config, str(args.run_num))
    else:
        expDir = os.path.join('/jizhicfs/Prometheus/gaoyuan/llm/ft_local/NeuralOA', params.exp_dir, args.config, str(args.run_num), args.finetune_dir)
    logging.info(f'expDir: {expDir}')
    params['experiment_dir']       = expDir 
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt.tar')

    # set up logging
    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference.log'))
    logging_utils.log_versions()
    params.log()

    if params["ics_type"] == 'default':
        num_samples = params.n_samples_per_year-params.prediction_length
        stop = num_samples
        # ics = np.arange(0, stop, params.decorrelation_time)
        ics = np.arange(0, 40, 1)
        n_ics = len(ics)
        print('init_condition:', ics)

    logging.info("Inference for {} initial conditions".format(n_ics))

    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""
    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    # get data and models
    valid_data_full, valid_data_full_ocean, model = setup(params)

    seq_pred = []
    seq_real = []

    for i, ic_ in enumerate(ics):
        logging.info("Initial condition {} of {}".format(i+1, n_ics))
        seq_real, seq_pred = autoregressive_inference(params, ic_, valid_data_full, valid_data_full_ocean, model)

        prediction_length = seq_real[0].shape[0]
        n_out_channels = seq_real[0].shape[1]
        img_shape_x = seq_real[0].shape[2]
        img_shape_y = seq_real[0].shape[3]

        # save predictions and loss
        save_path = os.path.join(params['experiment_dir'], 'results.h5')
        logging.info("Saving to {}".format(save_path))
        print(f'saving to {save_path}')
        if i == 0:
            f = h5py.File(save_path, 'w')
            f.create_dataset(
                    "ground_truth",
                    data=seq_real,
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.create_dataset(
                    "predicted",       
                    data=seq_pred, 
                    maxshape=[None, prediction_length, n_out_channels, img_shape_x, img_shape_y], 
                    dtype=np.float32)
            f.close()
        else:
            f = h5py.File(save_path, 'a')
            f["ground_truth"].resize((f["ground_truth"].shape[0] + 1), axis = 0)
            f["ground_truth"][-1:] = seq_real 

            f["predicted"].resize((f["predicted"].shape[0] + 1), axis = 0)
            f["predicted"][-1:] = seq_pred 

            f.close()

