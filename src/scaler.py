# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from info import train_years as years

import xarray as xr
import numpy as np
import logging
import glob
import os

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')
logging.info(f'Program started')

# data directories
data_dir = './data'
datasets_dir = os.path.join(data_dir, 'datasets', 'patches', 'train')
scaler_dir = os.path.join(data_dir, 'datasets', 'patches')
os.makedirs(scaler_dir, exist_ok=True)

# set the data pattern to retrieve the data from the disk
dataset_pattern_dir = os.path.join(datasets_dir, '*.zarr')

# define scaler filename
std_scaler_filename = f'std_{years[0]}_{years[-1]}.nc'
mean_scaler_filename = f'mean_{years[0]}_{years[-1]}.nc'
min_scaler_filename = f'min_{years[0]}_{years[-1]}.nc'
max_scaler_filename = f'max_{years[0]}_{years[-1]}.nc'
std_scaler_path = os.path.join(scaler_dir, std_scaler_filename)
mean_scaler_path = os.path.join(scaler_dir, mean_scaler_filename)
min_scaler_path = os.path.join(scaler_dir, min_scaler_filename)
max_scaler_path = os.path.join(scaler_dir, max_scaler_filename)

# get all the filenames in the directory
files = sorted(glob.glob(dataset_pattern_dir))

# define drivers to scale
drivers = ['fg10', 'i10fg', 'msl', 't_500', 't_300', 'vo_850', 'sst']

# log
logging.info(f'Opening zarr files')

data = np.empty(shape=(len(drivers), 0, 40, 40))
for file in files:
    # log
    logging.info(f'   {file}')
    x = xr.open_zarr(file)[drivers].to_array().load()
    data = np.concatenate((data, x), axis=1)

data_mean = np.nanmean(data, axis=(1,2,3))
data_std = np.nanstd(data, axis=(1,2,3))

# store to disk as xarray dataset
m = xr.load_dataset(os.path.join(scaler_dir, 'scaler_template.nc'))
s = xr.load_dataset(os.path.join(scaler_dir, 'scaler_template.nc'))

for i,drv in enumerate(drivers):
    m[drv] = data_mean[i]

for i,drv in enumerate(drivers):
    s[drv] = data_std[i]

# store to disk as netcdf
m.to_netcdf(mean_scaler_path)
s.to_netcdf(std_scaler_path)
