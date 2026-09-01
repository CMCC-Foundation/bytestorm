# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from mpi4py import MPI
import pandas as pd
import numpy as np
import logging
import os
import sys
sys.path.append('./resources/library')
from tropical_cyclone.diskio import TCDetectionMPIDatasetWriter
from tropical_cyclone._cyclone.macros import BASINS_COORDS
from tropical_cyclone.utils import build_rolling_months
from info import train_years, valid_years

# get basin name from 
basin = sys.argv[1]

# get lat and lon ranges from basin
lat_range, lon_range = BASINS_COORDS[basin]

# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# directories setup
main_dir = './'
data_dir = os.path.join(main_dir, 'data')
datasets_dir = os.path.join(data_dir, 'datasets')
src_data_dir = os.path.join(datasets_dir, 'global')
dst_patches_data_dir = os.path.join(datasets_dir, 'patches-global')
dst_train_data_dir = os.path.join(dst_patches_data_dir, f'{basin}', 'train')
dst_valid_data_dir = os.path.join(dst_patches_data_dir, f'{basin}', 'valid')
georef_src = os.path.join(data_dir, 'ibtracs', 'global_georef.csv')
patch_vars = ['msl', 'vo_850', 'density_map_tc']
coo_vars = ['real_cyclone', 'rounded_cyclone', 'global_cyclone', 'patch_cyclone']
log_dir = 'logs/build_dataset_patches'

# hyperparameters
dtype = np.float32
sigma = 5
grid_res = 0.25
patch_size = 40
label_no_cyclone = -1.0

# create directory if not exists
if not rank:
    os.makedirs(dst_train_data_dir, exist_ok=True)
    os.makedirs(dst_valid_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# load dataframes
georef = pd.read_csv(georef_src, index_col=0)
# fill nan (North Atlantic)
georef['BASIN'] = georef['BASIN'].fillna(value='NA')
# use only TCs coming from that basin
georef = georef[georef['BASIN']==basin]

# init the dataset writer
dataset_writer = TCDetectionMPIDatasetWriter(
    src_dir=src_data_dir,
    georef=georef,
    patch_vars=patch_vars,
    coo_vars=coo_vars,
    lat_range=lat_range,
    lon_range=lon_range,
    patch_size=patch_size,
    label_no_cyclone=label_no_cyclone,
    sigma=sigma,
    grid_res=grid_res,
    dtype=dtype,
)

# process the train dataset
_, train_years_dict = build_rolling_months(train_years[0], train_years[-1])
dataset_writer.process(dst_dir=dst_train_data_dir, years=None, months=None, years_dict=train_years_dict, is_test=False)
logging.info(f'Train dataset completed')

# process the valid dataset
_, valid_years_dict = build_rolling_months(valid_years[0], valid_years[-1])
dataset_writer.process(dst_dir=dst_valid_data_dir, years=None, months=None, years_dict=valid_years_dict, is_test=False)
logging.info(f'Valid dataset completed')

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()
