# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#
#                         Author : Davide Donno
#                    Download data directly from ERA5
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
from mpi4py import MPI
import xarray as xr
import pandas as pd
import logging
import os

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
log_dir = 'logs/gathering-global'

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
variables_single_level = ['mean_sea_level_pressure']
vorticity_variable = ['vorticity']
vorticity_pressure_level = [850]

# main directories setup
main_dir = './'
dataset_dir = os.path.join(main_dir, f'data/datasets/global')

# select only these ibtracs columns
columns = ['SID','SEASON','NUMBER','BASIN', 'SUBBASIN','NAME','ISO_TIME', 'NATURE', 'LAT','LON','WMO_WIND','WMO_PRES', 'TRACK_TYPE','DIST2LAND','LANDFALL', 'USA_WIND', 'USA_PRES', 'STORM_SPEED','STORM_DIR']

# open ibtracs dataset
ibtracs_src = os.path.join(main_dir, 'data/ibtracs/filtered/global_ibtracs_main-tracks_6h_1980-2022_TS-NR-ET-MX-SS-DS.csv')
ibtracs = pd.read_csv(ibtracs_src, usecols=columns, header=0, index_col=0, keep_default_na=False)

ibtracs['ISO_TIME'] = pd.to_datetime(ibtracs['ISO_TIME'])

# create folders if not exist
if rank == 0:
    os.makedirs(dataset_dir, exist_ok=True)

# barrier
comm.Barrier()

# get all the unique dates from ibtracs
dates = pd.to_datetime(sorted(ibtracs['ISO_TIME'].unique()))

logging.info(f'Loading Weatherbench...')

# load Weatherbench dataset
wb = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr')
logging.info(f'   subsetting dataset with dates')
wb = wb.sel(time=dates, latitude=slice(70,-70))
logging.info(f'   subsetting dataset with variables')
wb = wb[variables_single_level + vorticity_variable]
logging.info(f'   subsetting dataset with depth levels')
wb = wb.sel(level=vorticity_pressure_level)

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
        fname = os.path.join(dataset_dir, f'{year}_{month:02d}_{day:02d}_{hour:02d}.nc')

        if os.path.exists(fname):
            # log
            logging.info(f'File {date} already downloaded, testing consistency')
            # if file is consistent we can continue
            if is_consistent(path=fname):
                continue
            # otherwise we will download it again

        # download from WeatherBench2
        wbsub = wb.sel(time=slice(date,date)).compute()
        # build dataset as we need
        wbsub["vo_850"] = wbsub["vorticity"].sel(level=850)
        wbsub["msl"] = wbsub["mean_sea_level_pressure"]
        wbsub = wbsub.drop_vars(variables_single_level + vorticity_variable + ['level'])
        wbsub = wbsub.rename_dims({'latitude':'lat', 'longitude':'lon'}).rename_vars({'latitude':'lat', 'longitude':'lon'})

        # log
        logging.info(f'   Stored as NetCDF file for time {date}')

        # store the dataset to disk
        wbsub.to_netcdf(path=fname)

        # log
        logging.info(f'   File {date} successfully donwloaded')

        # check consistency of the downloaded sample
        is_consistent(fname)

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()
