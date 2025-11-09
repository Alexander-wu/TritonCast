import logging
import glob
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import Tensor
import h5py
import math
# import cv2
from my_utils.img_utils import reshape_fields



def get_data_loader(params, files_pattern, distributed, train):
    dataset = GetDataset(params, files_pattern, train)
    sampler = DistributedSampler(dataset, shuffle=train) if distributed else None
    # DistributedSampler: 
    #   Allocate a part of the dataset to each process/gpu, 
    #   and avoid duplication of data between different processes

    dataloader = DataLoader(dataset,
                            batch_size  = int(params.batch_size),
                            num_workers = params.num_data_workers,
                            shuffle     = False,  # (sampler is None),
                            sampler     = sampler if train else None,
                            drop_last   = True,
                            pin_memory  = True)  # pin_memory能加快内存的Tensor转义到GPU的显存的速度

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset


class GetDataset(Dataset):
    def __init__(self, params, location, train):
        self.params = params
        self.location = location
        self.train = train
        self.orography = params.orography
        self.normalize = params.normalize
        self.dt = params.dt  # 需要预测的时间步
        self.n_history = params.n_history # 输入的时间步
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.ocean_channels = np.array(params.ocean_channels)
        self.atmos_channels = np.array(params.atmos_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)

        self._get_files_stats()
        self.add_noise = params.add_noise if train else False
        self.fusion_3d_2d = params.fusion_3d_2d
        self.climate_mean = np.load('/apdcephfs_qy3/share_301734960/easyluwu/gy/data/coupled_0.25_23layers/climate_mean_s_t_ssh.npy', mmap_mode='r')


    # 获取文件统计信息
    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)

        with h5py.File(self.files_paths[0], 'r') as _f: 
            logging.info("Getting file stats from {}".format(self.files_paths[0]))

            # self.n_samples_per_year = _f['fields'].shape[0] - 1  
            self.n_samples_per_year = _f['fields'].shape[0] - self.params.multi_steps_finetune 

            # original image shape (before padding)
            self.img_shape_x = _f['fields'].shape[2] - 1 # just get rid of one of the pixels
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]

        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location,
                                                                                                       self.n_samples_total,
                                                                                                       self.img_shape_x,
                                                                                                       self.img_shape_y,
                                                                                                       self.n_in_channels))
        logging.info("Delta t: {} days".format(1 * self.dt))
        logging.info("Including {} days of past history in training at a frequency of {} days".format(
            1 * self.dt * self.n_history, 1 * self.dt))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields'] 

        if self.orography and self.params.normalization == 'zscore': 
            _orog_file = h5py.File(self.params.orography_norm_zscore_path, 'r')
        if self.orography and self.params.normalization == 'maxmin': 
            _orog_file = h5py.File(self.params.orography_norm_maxmin_path, 'r')

    def __len__(self):
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx  = int(global_idx / self.n_samples_per_year)  # which year
        local_idx = int(global_idx % self.n_samples_per_year)  # which sample in a year

        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # If there are not enough historical time steps available in the features, shift to future time steps.
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # If the sample is the final one for the year, predict the current time step. Otherwise, predict the next time step.

        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.orography:
            orog = self.orography_field 
            if np.shape(orog)[0] == 721:
                orog = orog[0:720]
            # logging.info(f'orog: {orog.shape}')
        else:
            orog = None
        
        # logging.info(f'year_idx: {year_idx}, local_idx:{local_idx}, dt:{self.dt}, step:{step}, n_history:{self.n_history}')


        if self.params.multi_steps_finetune == 1:
            if local_idx == 365:
                local_idx = 364
            
            climate_mean_ocean = self.climate_mean[(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.ocean_channels, :720, :1440]
            ocean = reshape_fields( 
                    self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.ocean_channels, :720, :1440] - climate_mean_ocean, 
                    'ocean', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )
            
            climate_mean_force_future = self.climate_mean[local_idx+step, self.atmos_channels, :720, :1440]
            force_future = reshape_fields( 
                    self.files[year_idx][local_idx+step, self.atmos_channels, :720, :1440] - climate_mean_force_future, 
                    'force', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )

            # inp = reshape_fields( 
            #         self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.in_channels, :720, :1440], 
            #         'inp', 
            #         self.params, 
            #         self.train, 
            #         self.normalize, 
            #         orog, 
            #         self.add_noise 
            #     )
            climate_mean_tar = self.climate_mean[local_idx+step, self.out_channels, :720, :1440]
            tar = reshape_fields(
                    self.files[year_idx][local_idx+step, self.out_channels, :720, :1440] - climate_mean_tar, 
                    'tar', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )
        else:
            climate_mean_ocean = self.climate_mean[(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.ocean_channels, :720, :1440]
            ocean = reshape_fields( 
                    self.files[year_idx][(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, self.ocean_channels, :720, :1440] - climate_mean_ocean, 
                    'ocean', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )

            climate_mean_force_future = self.climate_mean[local_idx+step, self.atmos_channels, :720, :1440]
            force_future = reshape_fields( 
                    self.files[year_idx][local_idx+step, self.atmos_channels, :720, :1440] - climate_mean_force_future, 
                    'force', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog, 
                    self.add_noise 
                )

            
            climate_mean_tar = self.climate_mean[local_idx+step:local_idx+step+self.params.multi_steps_finetune, self.in_channels, :720, :1440]
            tar_data = self.files[year_idx][local_idx+step:local_idx+step+self.params.multi_steps_finetune, self.in_channels, :720, :1440]
            tar = reshape_fields( 
                    tar_data - climate_mean_tar, 
                    'inp', 
                    self.params, 
                    self.train, 
                    self.normalize, 
                    orog 
                )
            
        ocean = np.nan_to_num(ocean, nan=0)
        force_future = np.nan_to_num(force_future, nan=0)
        # inp = np.nan_to_num(inp, nan=0)
        tar = np.nan_to_num(tar, nan=0)

        # print('ocean.shape, force_future.shape, tar.shape', ocean.shape, force_future.shape, tar.shape)

        return np.concatenate((ocean, force_future), axis=0), tar 
        # return inp, tar  





