from tropical_cyclone.cyclone import retrieve_predicted_tc, ERA5_CMIP6_DRIVERS_MAPPING, init_track_dataframe, tracking_algorithm, mean, std
from tropical_cyclone.utils import round_to_grid
from tropical_cyclone.scaling import StandardScaler
from tropical_cyclone.models import *

from torch.utils.data import TensorDataset, DataLoader, Dataset
from scipy.ndimage import measurements
from einops import rearrange
from tqdm import tqdm
import torch.nn as nn
import xarray as xr
import pandas as pd
import numpy as np
import logging
import joblib
import torch
import munch
import glob
import toml
import os

def load_trained_model(model_dir, device='cpu'):
    # read config file
    config = munch.munchify(toml.load(os.path.join(model_dir, 'configuration.toml')))
    # get model weights file
    model_weights_file = sorted(glob.glob(os.path.join(model_dir, 'checkpoints', '*.ckpt')))[-1]
    # log
    logging.info(f'Loading model checkpoint at {model_weights_file}')
    # init model class from config
    model_cls = eval(config.model.cls)
    # get model arguments from config
    model_args = config.model.args
    # create the model
    model:nn.Module = model_cls(**model_args)
    # load state dict
    model_state_dict = torch.load(f=model_weights_file, map_location=device)
    # load weights into model
    model.load_state_dict(model_state_dict['state_dict'])
    # put the model to device
    model.to(device)
    # put the model in eval mode
    model.eval()
    return model, config, model_weights_file

def prepare_dataloader(patch_ds, patch_size, scaler, drivers, time, rows, cols, channels, batch_size=4096):
    # load dataset to numpy
    x = patch_ds[drivers].to_array().load().data
    # transpose drivers to last channel
    x = np.transpose(x, axes=(1,2,3,4,5,0))
    # put rows and cols channels near time dimension
    x = np.transpose(x, axes=(0,1,3,2,4,5))
    # reshape aggregating time, rows and cols
    x = np.reshape(x, newshape=(time*rows*cols, patch_size, patch_size, channels))
    # transform the data with the scaler
    x = scaler.transform(torch.as_tensor(x))
    # remove all nan values (if present)
    x = np.nan_to_num(x, nan=0.0)
    # move drivers to second dimension
    x = (np.transpose(x, axes=(0,3,1,2)))
    # convert to dataset
    dataset = TensorDataset(torch.as_tensor(x, dtype=torch.float32))
    # create a dataloader
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return data_loader

def predict_with_models(models, data_loader, targets = None, patch_size = None, device = 'cpu'):
    if type(models) != list: models = [models]
    models_predictions = []
    for model in models:
        # predict with the trained model
        if targets is not None and patch_size is not None:
            y_pred = np.empty(shape=(0, len(targets), patch_size, patch_size))
        else:
            y_pred = np.empty(shape=(0,2))
        for data in tqdm(data_loader):
            # x = data[0].to(device)
            x = data.to(device)
            x = rearrange(x, 'b1 b2 d h w -> (b1 b2) d h w')
            y_pred = np.concatenate([y_pred, model(x).cpu().detach().numpy()])
        # reshape disgregating time, rows and cols
        models_predictions.append(y_pred)
    # stack together each model's predictions
    models_predictions = np.stack(models_predictions, axis=1)
    if models_predictions.shape[1] == 1: models_predictions = models_predictions[:,0]
    return models_predictions

def predict_with_models_clsloc(models, data_loader, targets = None, patch_size = None, device = 'cpu'):
    if type(models) != list: models = [models]
    models_predictions = []
    for model in models:
        # predict with the trained model
        if targets is not None and patch_size is not None:
            y_pred = np.empty(shape=(0, len(targets), patch_size, patch_size))
        else:
            y_pred = np.empty(shape=(0,2))
        for data in tqdm(data_loader):
            x = data[0].to(device)
            y_hat_loc, y_hat_cls = model(x)
            y_hat_loc, y_hat_cls = y_hat_loc.cpu().detach().numpy(), y_hat_cls.cpu().detach().numpy()
            y_hat_loc[np.where(y_hat_cls < 0.99)[0]] = -1.0
            y_pred = np.concatenate([y_pred, y_hat_loc])
        # reshape disgregating time, rows and cols
        models_predictions.append(y_pred)
    # stack together each model's predictions
    models_predictions = np.stack(models_predictions, axis=1)
    if models_predictions.shape[1] == 1: models_predictions = models_predictions[:,0]
    return models_predictions

def predict_with_models_cls(models, data_loader, device = 'cpu'):
    if type(models) != list: models = [models]
    models_predictions = []
    for model in models:
        # predict with the trained model
        y_pred = np.empty(shape=(0,1))
        for data in tqdm(data_loader):
            # x = data[0].to(device)
            x = data.to(device)
            x = rearrange(x, 'b1 b2 d h w -> (b1 b2) d h w')
            y_pred = np.concatenate([y_pred, model(x).cpu().detach().numpy()])
        # reshape disgregating time, rows and cols
        models_predictions.append(y_pred)
    # stack together each model's predictions
    models_predictions = np.stack(models_predictions, axis=1)
    if models_predictions.shape[1] == 1: models_predictions = models_predictions[:,0]
    return models_predictions

def get_detections(patch_ds: xr.Dataset):
    if 'patch_cyclone_probability' in patch_ds.variables:
        variables = ['patch_cyclone_pred', 'patch_cyclone_probability']
        on_cols = ['time', 'rows', 'cols', 'patch_cyclone_probability']
        sel_cols =['time', 'patch_cyclone_pred_x', 'patch_cyclone_pred_y', 'patch_cyclone_probability']
        rename_cols = {'patch_cyclone_pred_x':'LAT', 'patch_cyclone_pred_y':'LON', 'patch_cyclone_probability': 'PROB'}
    else:
        variables = 'patch_cyclone_pred'
        on_cols = ['time', 'rows', 'cols']
        sel_cols =['time', 'patch_cyclone_pred_x', 'patch_cyclone_pred_y']
        rename_cols = {'patch_cyclone_pred_x':'LAT', 'patch_cyclone_pred_y':'LON'}
    # convert cyclone coordinates to pandas dataframe
    df = patch_ds[variables].to_dataframe().reset_index()
    # merge dataframe to get coordinates
    on_cols
    detections = pd.merge(left=df[df['coordinate']==0], right=df[df['coordinate']==1], on=on_cols)
    # take only time and coordinates
    detections = detections[sel_cols]
    # rename coordinates to LAT and LON
    detections = detections.rename(columns=rename_cols)
    # remove NaN rows
    detections = detections[~(np.isnan(detections['LAT']) & np.isnan(detections['LON']))]
    # convert time axis to datetime
    detections['time'] = pd.to_datetime(detections['time'].astype(str))
    # convert proj and era5 `time` col to `ISO_TIME`
    detections = detections.rename(columns={'time':'ISO_TIME'})
    # add infinite wind speed on each row
    detections['WS'] = np.inf
    # reset index
    detections = detections.reset_index(drop=True)
    return detections


class Inference():
    def __init__(self, device='cpu') -> None:
        self.device = device

    def predict(self):
        raise NotImplementedError
    
    def store_detections(self, detections, dst):
        if detections is None:
            logging.info(f'No detections found')
            return
        if os.path.exists(dst):
            logging.info(f'File already existing at {dst}')
            return
        # store detections to disk
        detections.to_csv(dst)
        logging.info(f'Detections stored at {dst}')

    def _parse_config_file(self, config):
        drivers = config.data.drivers
        targets = config.data.targets
        scaler = StandardScaler(mean_src=config.dir.scaler.mean, std_src=config.dir.scaler.std, drivers=drivers)
        return scaler, drivers, targets

    def load_dataset(self, dataset_dir, drivers, year=None, is_cmip6=False):
        if is_cmip6:
            drivers = [ERA5_CMIP6_DRIVERS_MAPPING[d] for d in drivers]
        if year is not None: pattern = f'{year}*.nc'
        else: pattern = f'*.nc'
        files = sorted(glob.glob(os.path.join(dataset_dir, pattern)))
        logging.info(f'Opening dataset containing {len(files)} files')
        if len(files) == 0: return None
        ds = xr.open_mfdataset(files)
        logging.info(f'Dataset opened')
        if ds is None:
            logging.info(f'No dataset found')
            return None
        dates = pd.to_datetime(ds['time'].astype(str))
        return ds, dates

    def load_dataset_by_hemisphere(self, 
            dataset_dir: str,
            georef: pd.DataFrame,
            hemisphere: str,
            year: int = None,
        ):
        # make a copy of the georef
        gr = georef.copy()

        if hemisphere == 'north':
            gr = gr[gr['LAT'] >= 0]
            lat_slice = slice(70,0)
        elif hemisphere == 'south':
            gr = gr[gr['LAT'] <= 0]
            lat_slice = slice(0,-70)

        # convert georef iso time to pandas datetime
        gr['ISO_TIME'] = pd.to_datetime(gr['ISO_TIME'])

        # select only TCs belonging to the passed year
        gr = gr[gr['ISO_TIME'].dt.year==year]

        # read the dates
        dates = gr['ISO_TIME'].unique()

        # create the files list
        files = []
        for date in dates:
            files.append(os.path.join(dataset_dir, f'{date.year}_{date.month:02d}_{date.day:02d}_{date.hour:02d}.nc'))

        logging.info(f'Opening dataset containing {len(files)} files')
        if len(files) == 0: return None

        # open the dataset files
        ds = xr.open_mfdataset(files)
        logging.info(f'Dataset opened')

        if ds is None:
            logging.info(f'No dataset found')
            return None

        # get only the selected hemisphere
        ds = ds.sel(lat=lat_slice)
        logging.info(f'Dataset subset')

        dates = pd.to_datetime(ds['time'].astype(str))
        return ds, dates



class SingleModelInference(Inference):
    def __init__(self, model_dir, device='cpu') -> None:
        super().__init__(device)
        self.model_dir = model_dir
        self.model, self.config, _ = load_trained_model(model_dir, device)
        self.scaler, self.drivers, self.targets = self._parse_config_file(self.config)

    def predict(self, ds: xr.Dataset,         # xarray dataset containing input data
                      patch_size: int = 40,   # dimension of a patch
                      eps: float = 0.1,       # tolerance value for negative label (ONLY regression task)
                      roll: bool = False,     # whether to roll the data (ONLY regression task)
                      threshold: float = 9.5, # the threshold value to consider the density a TC (ONLY map task)
                      classify: bool = False, # whether the model predicts the classification
    ):
        lons = ds['lon'].shape[0]
        lats = ds['lat'].shape[0]
        rows = lats // patch_size
        cols = lons // patch_size
        time, channels = ds['time'].shape[0], len(self.drivers)
        if roll:
            # roll the dataset by half of patch size
            ds = ds.roll(shifts={'lat':-patch_size//2, 'lon':-patch_size//2}, roll_coords=True)
        # divide dataset in patches
        patch_ds = ds.coarsen({'lat':patch_size, 'lon':patch_size}, boundary="trim").construct({'lon':("cols", "lon_range"), 'lat':("rows", "lat_range")})
        if roll:
            # remove rolled patches as there cannot be a TC within those patches
            patch_ds = patch_ds.isel(rows=slice(0,-1), cols=slice(0,-1))
            rows -= 1
            cols -= 1
        # get dataloader
        data_loader = prepare_dataloader(patch_ds=patch_ds, patch_size=patch_size, scaler=self.scaler, drivers=self.drivers, 
                                         time=time, rows=rows, cols=cols, channels=channels, batch_size=4096)
        # predict
        if 'density_map_tc' in self.targets: # it is the map approach
            y_pred = predict_with_models(models=[self.model], data_loader=data_loader, targets=self.targets, patch_size=patch_size, device=self.device)
            filt_y_pred = np.where(y_pred >= threshold, 1.0, 0.0)
            b = 256
            tc_center_estimates = np.empty(shape=(0, 3), dtype=np.int32)
            for i in range(0, filt_y_pred.shape[0], b):
                for j in range(filt_y_pred.shape[1]):
                    tc_objects, n_clusters = measurements.label(filt_y_pred[i:i+b,j])
                    if n_clusters == 0: continue
                    tc_estimates = np.round(np.stack([np.median(np.where(tc_objects == i), axis=1) for i in range(1, n_clusters+1)])).astype(np.int32)
                    tc_estimates[:,0] = i + tc_estimates[:,0]
                    tc_center_estimates = np.concatenate((tc_center_estimates, tc_estimates))
            y_pred = np.full(shape = (time * rows * cols, 2), fill_value = -1.0)
            y_pred[tc_center_estimates[:,0]] = tc_center_estimates[:,-2:]
        else:
            if classify:
                y_pred = predict_with_models_clsloc(models=[self.model], data_loader=data_loader, device=self.device)
            else:
                y_pred = predict_with_models(models=[self.model], data_loader=data_loader, device=self.device)
        y_pred = np.reshape(y_pred, newshape=(time, rows, cols, 2))
        # get predicted cyclone coordinates
        patch_ds = retrieve_predicted_tc(y_pred, ds, patch_ds, patch_size, eps=eps)
        # get detections
        detections = get_detections(patch_ds)
        return detections


class TCDataset(Dataset):
        def __init__(self, patch_ds: xr.Dataset, patch_size: int, drivers: list, scaler: any):
            super().__init__()
            self.scaler = scaler
            self.drivers = drivers
            self.patch_ds = patch_ds
            self.patch_size = patch_size
        def __len__(self):
            return self.patch_ds.time.shape[0]
        def __getitem__(self, i: int):
            x = self.patch_ds.isel(time=i)[self.drivers].to_array().values # D x R x 40 x C x 40
            x = torch.as_tensor(x)
            x = rearrange(x, 'd r p1 c p2 -> (r c) p1 p2 d')
            x = self.scaler.transform(x)
            x = torch.nan_to_num(x, nan=0.0)
            x = rearrange(x, 'k p1 p2 d -> k d p1 p2')
            return x

def prepare_dataloader_new(dataset: TCDataset, batch_size: int = 1, num_workers: int = 1):
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

import time as timex

class LocClsModelInference(Inference):
    def __init__(self, model_loc_dir, model_cls_dir, device='cpu') -> None:
        super().__init__(device)
        self.model_loc_dir = model_loc_dir
        self.model_cls_dir = model_cls_dir
        self.model_loc, self.config_loc, _ = load_trained_model(model_loc_dir, device)
        self.model_cls, self.config_cls, _ = load_trained_model(model_cls_dir, device)
        self.scaler_loc, self.drivers_loc, self.targets_loc = self._parse_config_file(self.config_loc)
        self.scaler_cls, self.drivers_cls, self.targets_cls = self._parse_config_file(self.config_cls)

    def predict(self, 
            ds: xr.Dataset,          # xarray dataset containing input data
            patch_size: int = 40,    # dimension of a patch
            threshold: float = 0.5,  # threshold for classification
            batch_size: int = 1, 
            num_workers: int = 1, 
        ):
        lons = ds['lon'].shape[0]
        lats = ds['lat'].shape[0]
        rows = lats // patch_size
        cols = lons // patch_size
        time, channels_loc, channels_cls = ds['time'].shape[0], len(self.drivers_loc), len(self.drivers_cls)
        # divide dataset in patches
        patch_ds = ds.coarsen({'lat':patch_size, 'lon':patch_size}, boundary="trim").construct({'lon':("cols", "lon_range"), 'lat':("rows", "lat_range")})
        # get dataloader
        # data_loader_loc = prepare_dataloader(
        #     patch_ds = patch_ds, 
        #     patch_size = patch_size, 
        #     scaler = self.scaler_loc, 
        #     drivers = self.drivers_loc, 
        #     time = time, rows=rows, cols=cols, 
        #     channels = channels_loc, 
        #     batch_size = 4096)
        # data_loader_cls = prepare_dataloader(
        #     patch_ds = patch_ds, 
        #     patch_size = patch_size, 
        #     scaler = self.scaler_cls, 
        #     drivers = self.drivers_cls, 
        #     time = time, rows=rows, cols=cols, 
        #     channels = channels_cls, 
        #     batch_size = 4096)
        exec_time = -timex.time()
        dataset_loc = TCDataset(patch_ds, patch_size, self.drivers_loc, self.scaler_loc)
        dataset_cls = TCDataset(patch_ds, patch_size, self.drivers_cls, self.scaler_cls)
        data_loader_loc = prepare_dataloader_new(dataset_loc, batch_size, num_workers)
        data_loader_cls = prepare_dataloader_new(dataset_cls, batch_size, num_workers)
        # predict
        y_pred_loc = predict_with_models(
            models = [self.model_loc], data_loader = data_loader_loc, device = self.device
            )
        y_pred_cls = predict_with_models_cls(
            models = [self.model_cls], data_loader = data_loader_cls, device = self.device
            )
        exec_time += timex.time()
        print(f'Data Load + Inference Execution Time {exec_time} seconds')
        # reshape and filter the data
        y_pred_cls_thr = np.where(y_pred_cls <= threshold, 0.0, 1.0)
        y_pred_loc[(y_pred_cls_thr == 0)[:,0]] = -1.0
        y_pred_cls_thr = y_pred_cls_thr.reshape((time, rows, cols, -1))
        y_pred_cls = y_pred_cls.reshape((time, rows, cols, -1))
        y_pred_loc = y_pred_loc.reshape((time, rows, cols, 2))
        # get predicted cyclone coordinates
        patch_ds = retrieve_predicted_tc(y_pred_loc, ds, patch_ds, patch_size, eps=0.1, y_pred_cls=y_pred_cls)
        # get detections
        detections = get_detections(patch_ds)
        return detections

    def predict_old(self, 
            ds: xr.Dataset,          # xarray dataset containing input data
            patch_size: int = 40,    # dimension of a patch
            threshold: float = 0.5,  # threshold for classification
        ):
        lons = ds['lon'].shape[0]
        lats = ds['lat'].shape[0]
        rows = lats // patch_size
        cols = lons // patch_size
        time, channels_loc, channels_cls = ds['time'].shape[0], len(self.drivers_loc), len(self.drivers_cls)
        # divide dataset in patches
        patch_ds = ds.coarsen({'lat':patch_size, 'lon':patch_size}, boundary="trim").construct({'lon':("cols", "lon_range"), 'lat':("rows", "lat_range")})
        # get dataloader
        data_loader_loc = prepare_dataloader(
            patch_ds = patch_ds, 
            patch_size = patch_size, 
            scaler = self.scaler_loc, 
            drivers = self.drivers_loc, 
            time = time, rows=rows, cols=cols, 
            channels = channels_loc, 
            batch_size = 4096)
        data_loader_cls = prepare_dataloader(
            patch_ds = patch_ds, 
            patch_size = patch_size, 
            scaler = self.scaler_cls, 
            drivers = self.drivers_cls, 
            time = time, rows=rows, cols=cols, 
            channels = channels_cls, 
            batch_size = 4096)
        # predict
        y_pred_loc = predict_with_models(
            models = [self.model_loc], data_loader = data_loader_loc, device = self.device
            )
        y_pred_cls = predict_with_models_cls(
            models = [self.model_cls], data_loader = data_loader_cls, device = self.device
            )
        # reshape and filter the data
        y_pred_cls_thr = np.where(y_pred_cls <= threshold, 0.0, 1.0)
        y_pred_loc[(y_pred_cls_thr == 0)[:,0]] = -1.0
        y_pred_cls_thr = y_pred_cls_thr.reshape((time, rows, cols, -1))
        y_pred_cls = y_pred_cls.reshape((time, rows, cols, -1))
        y_pred_loc = y_pred_loc.reshape((time, rows, cols, 2))
        # get predicted cyclone coordinates
        patch_ds = retrieve_predicted_tc(y_pred_loc, ds, patch_ds, patch_size, eps=0.1, y_pred_cls=y_pred_cls)
        # get detections
        detections = get_detections(patch_ds)
        return detections
