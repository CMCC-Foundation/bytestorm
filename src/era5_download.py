# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from mpi4py import MPI
import xarray as xr
import pandas as pd
import logging
import cdsapi
import shutil
import os

import sys
sys.path.append('./resources/library/tropical_cyclone')
from tropical_cyclone.era5 import retrieve_era5_pressure_levels, retrieve_era5_single_levels

def is_consistent(path):
    try:
        # test consistency of the dataset
        ds = xr.open_mfdataset(path)
        ds.close()
        return True
    except:
        # remove inconsistent file
        os.remove(path)
        # log
        logging.info(f'   Downloaded file is not consistent. Removed.')
        return False


# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# define log directory
log_dir = 'logs'

# only rank 0 will create the folders
if not rank:
    os.makedirs(log_dir, exist_ok=True)

# wait for all the processes to arrive here
comm.Barrier()

# initialize logger 
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", filename=f"{log_dir}/proc-{rank}.log", 
                    filemode="w", level=logging_level, datefmt='%Y-%m-%d %H:%M:%S')

# log
logging.info(f'Execution started')

# define download variables
variables_single_level = ['instantaneous_10m_wind_gust', 'mean_sea_level_pressure', 'sea_surface_temperature']
wind_post_processing = ['10m_wind_gust_since_previous_post_processing']
vorticity_variable = ['vorticity']
vorticity_pressure_level = ['850']
temperature_variable = ['temperature']
temperature_pressure_levels = ['300','500']

# domain extent
south = 0 # °N
north = 70 # °N
west = 100 # °E
east = -40 # °E

# main directories setup
in_dataset_dir = f'../../data/datasets/north_atlantic'
out_dataset_dir = f'../../data/datasets/temp_tc_dataset'
dataset_tmp_dir = f'../../data/datasets/tmp'
dataset_tmp_data_dir = os.path.join(dataset_tmp_dir, f'data_{rank}')
dataset_tmp_wind_gust_dir = os.path.join(dataset_tmp_dir, f'wind_gust_{rank}')

# select only these ibtracs columns
columns = ['SID','SEASON','NUMBER','BASIN', 'SUBBASIN','NAME','ISO_TIME', 'NATURE', 'LAT','LON','WMO_WIND','WMO_PRES', 'TRACK_TYPE','DIST2LAND','LANDFALL', 'USA_WIND', 'USA_PRES', 'STORM_SPEED','STORM_DIR']

# open ibtracs dataset
ibtracs_src = '../../data/ibtracs/filtered/north_atlantic_ibtracs_main-tracks_6h_1980-2021_TS-NR-ET-MX-SS-DS.csv'
ibtracs = pd.read_csv(ibtracs_src, usecols=columns, header=0, index_col=0, keep_default_na=False)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#  Program Start
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# init CDS-API Client
client = cdsapi.Client()

# create folders if not exist
if rank == 0:
    os.makedirs(in_dataset_dir, exist_ok=True)
    os.makedirs(out_dataset_dir, exist_ok=True)

# remove temporary directory files
shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)

# create temporary directory
if rank == 0:
    os.makedirs(dataset_tmp_data_dir, exist_ok=True)
    os.makedirs(dataset_tmp_wind_gust_dir, exist_ok=True)

# barrier
comm.Barrier()

# get all the unique dates from ibtracs
dates = pd.to_datetime(sorted(ibtracs['ISO_TIME'].unique()))

# iterate over each iso time
for i,date in enumerate(dates):

    # download only if rank is equal to i % size
    if i % size == rank:
        year = date.year
        month = date.month
        day = date.day
        hour = date.hour

        logging.info(f'Downloading ERA5 variable maps for iso time {year}-{month}-{day} {hour}:00')

        # define output filename
        in_fname = os.path.join(in_dataset_dir, f'{year}_{month:02d}_{day:02d}_{hour:02d}.nc')
        out_fname = os.path.join(out_dataset_dir, f'{year}_{month:02d}_{day:02d}_{hour:02d}.nc')

        if os.path.exists(in_fname):
            # log
            logging.info(f'File {date} already downloaded, testing consistency')
            # if file is consistent we can continue
            if is_consistent(path=in_fname):
                continue
            # otherwise we will download it again

        # download era5 single levels variables
        try:
            retrieve_era5_single_levels(client=client, out_dir=dataset_tmp_data_dir, variables=variables_single_level, south=south, north=north, west=west, east=east, year=year, month=month, day=day, hour=hour)
        except Exception as e:
            logging.info(f'An exception occurred during download at time {date}. Error : {e}. Skipping')
            shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
            shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)
            continue

        # download wind gust since previous post-processing
        for i in range(6):
            try:
                prev_date = date - pd.DateOffset(hours=i)
                retrieve_era5_single_levels(client=client, out_dir=dataset_tmp_wind_gust_dir, variables=wind_post_processing, south=south, north=north, west=west, east=east, year=year, month=month, day=day, hour=hour)
                wind_post_processing_download_failed = False
            except Exception as e:
                logging.info(f'An exception occurred during download at time {date}. Error : {e}. Skipping')
                shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
                shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)
                wind_post_processing_download_failed = True
                break

        # check if wind gust has been successfully downloaded
        if wind_post_processing_download_failed:
            logging.info(f'Could not download wind gust. Skipping')
            continue

        # download era5 temperature variables
        try:
            retrieve_era5_pressure_levels(client=client, out_dir=dataset_tmp_data_dir, variables=temperature_variable, pressure_levels=temperature_pressure_levels, south=south, north=north, west=west, east=east, year=year, month=month, day=day, hour=hour)
        except Exception as e:
            logging.info(f'An exception occurred during download at time {date}. Error : {e}. Skipping')
            shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
            shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)
            continue

        # download era5 vorticity variable
        try:
            retrieve_era5_pressure_levels(client=client, out_dir=dataset_tmp_data_dir, variables=vorticity_variable, pressure_levels=vorticity_pressure_level, south=south, north=north, west=west, east=east, year=year, month=month, day=day, hour=hour)
        except Exception as e:
            logging.info(f'An exception occurred during download at time {date}. Error : {e}. Skipping')
            shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
            continue

        # open wind gust dataset from disk
        wind_gust_ds = xr.open_mfdataset(os.path.join(dataset_tmp_wind_gust_dir, '*.nc')).load()

        # resample at 6 hours
        wind_gust_resample = wind_gust_ds.resample(time="6H", label='right', closed='right')

        # max of resample
        max_wind_gust_ds = wind_gust_resample.max(dim='time', keep_attrs=True)

        # open all the data that we just downloaded
        data_ds = xr.open_mfdataset(os.path.join(dataset_tmp_data_dir, '*.nc'))

        # merge max wind gust resample with other dataset
        data_ds = xr.merge([data_ds, max_wind_gust_ds], combine_attrs='override')

        # remove temperature pressure level
        for level_xr in data_ds['level']:
            level = level_xr.data
            data_ds[f't_{level}'] = (('time','latitude','longitude'), data_ds['t'].sel(level=level).data)
            data_ds[f't_{level}'].attrs = data_ds['t'].attrs
        data_ds = data_ds.drop('t').drop_dims('level')

        # rename latitude and longitude dimensions
        data_ds = data_ds.rename_dims({'latitude':'lat', 'longitude':'lon'})

        # save dataset to disk
        data_ds.to_netcdf(out_fname)

        # remove temporary directory files
        shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
        shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)

        # re-create temporary directory
        os.makedirs(dataset_tmp_data_dir, exist_ok=True)
        os.makedirs(dataset_tmp_wind_gust_dir, exist_ok=True)

        # log
        logging.info(f'Dataset correctly saved to disk')

        # check consistency of the downloaded sample
        is_consistent(out_fname)

        break

# remove temporary data, if present
shutil.rmtree(dataset_tmp_data_dir, ignore_errors=True)
shutil.rmtree(dataset_tmp_wind_gust_dir, ignore_errors=True)

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()
