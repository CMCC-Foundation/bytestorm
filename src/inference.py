# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from mpi4py import MPI
import pandas as pd
import logging
import os
import sys

import warnings
warnings.filterwarnings('ignore')

sys.path.append('./resources/library/tropical_cyclone')
from info import test_years
from tropical_cyclone.cyclone import LocClsModelInference


# initialize MPI
comm = MPI.COMM_WORLD

# get rank and world size of the process
rank = comm.Get_rank()
size = comm.Get_size()

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Program Parameters
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# main directories
experiment_dir = 'path-to/experiments'
data_dir = 'path-to/data'

experiment_loc = sys.argv[1]
experiment_cls = sys.argv[2]
threshold = float(sys.argv[3])

# select model
model_loc_dir = os.path.join(experiment_dir, f'{experiment_loc}')
model_cls_dir = os.path.join(experiment_dir, f'{experiment_cls}')
dataset_dir = os.path.join(data_dir, 'datasets/reanalysis')

# define inference folder
inference_dir = os.path.join(data_dir, 'inference')
inference_model_dir = os.path.join(inference_dir, os.path.basename(model_loc_dir) + '-' + os.path.basename(model_cls_dir))

# define logs directory
log_dir = f'logs_{experiment_loc}-{experiment_cls}'

# create directory if not exist
if not rank:
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(inference_dir, exist_ok=True)
    os.makedirs(inference_model_dir, exist_ok=True)
# wait for all the processes to arrive here
comm.Barrier()

# initialize logger
logging_level = logging.INFO
logging.basicConfig(format="[%(asctime)s] %(levelname)s : %(message)s", 
                    filename=f"{log_dir}/proc-{rank}.log", 
                    filemode="w", 
                    level=logging_level, 
                    datefmt='%Y-%m-%d %H:%M:%S')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#  Inference on the Dataset
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

logging.info(f'Starting inference')
logging.info(f'   selected loc model at {model_loc_dir}')
logging.info(f'   selected cls model at {model_cls_dir}')

inference = LocClsModelInference(model_loc_dir=model_loc_dir, model_cls_dir=model_cls_dir, device='cuda')
drivers_loc = inference.drivers_loc
drivers_cls = inference.drivers_cls
drivers = drivers_loc + drivers_cls

logging.info(f'Evaluating test set years')

for i in range(rank, len(test_years), size):
    year = test_years[i]
    logging.info(f'Year {year}')
    # get detection destination
    detection_dst = os.path.join(inference_model_dir, f'{year}.csv')
    logging.info(f'  Predicting...')
    # predict with the model
    ds, dates = inference.load_dataset(dataset_dir=dataset_dir, drivers=drivers, year=year, is_cmip6=False)
    detections = inference.predict(ds, patch_size=40, threshold=threshold)
    logging.info(f'   ...done')
    # store detections on disk
    inference.store_detections(detections, detection_dst)
    logging.info(f'   predictions stored')

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# wait for all the processes to finish
comm.Barrier()

# log
logging.info(f'Process completed')

# finalize the processes
MPI.Finalize()