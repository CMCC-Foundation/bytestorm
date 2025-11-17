# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

from tropical_cyclone.cyclone import (
    coo_rot180, 
    coo_left_right, 
    coo_up_down, 
    rot180, 
    left_right, 
    up_down
)
from tropical_cyclone.scaling import Scaler

from torch.utils.data import Dataset
from typing import List, Any
import xarray as xr
import logging
import torch
import glob
import os


def read_zarrs_as_torch_tensor(
        zarrs: List[xr.Dataset], 
        variables: List[str], 
        dtype = torch.float32
    ):
    data = []
    for zarr in zarrs:
        x = torch.as_tensor(zarr[variables].to_array().load().data, dtype=dtype)
        if len(x.shape) == 4:
            x = torch.permute(x, dims=(1,0,2,3))
        elif len(x.shape) == 3:
            x = torch.permute(x, dims=(1,0,2))
        data.append(x)
    return torch.concat(data, dim=0)


class TCPatchDataset(Dataset):
    def __init__(self, 
                 src: str,                               # source directory where the zarr dataset are stored
                 drivers: List[str],                     # drivers variable list
                 targets: List[str],                     # target variable list
                 scaler: Scaler = None,                  # scaler for the data
                 label_no_cyclone: float = -1.0,         # label assigned to NoCyclone (only for regression of TC coordinates)
                 mag: float = 1.0,                       # magnitude assigned to the output (only for density map prediction)
                 augmentation: list = ['lr','ud','rot'], # whether to augment the dataset (only TCs are augmented)
                 classify: bool = False,                 # whether to use the classification output
                 patches: list = ['cy','nr','rn'],       # the selected patches that we want to use
                 dtype = torch.float32,                  # datatype of tensors
                 ) -> None:
        super().__init__()
        # store params
        self.label_no_cyclone = label_no_cyclone
        self.augmentation = augmentation
        self.scaler: Scaler = scaler
        self.classify = classify
        self.dtype = dtype
        self.mag = mag
        self.patches = patches
        cy_zarrs, nr_zarrs, rn_zarrs = None, None, None
        cy_n, nr_n, rn_n = 0, 0, 0 # init to 0 the number of elements
        if 'cy' in patches:
            cy_files = sorted(glob.glob(os.path.join(src,'cyclone*.zarr'))) # get dataset filenames
            cy_zarrs = [xr.open_zarr(file) for file in cy_files]            # open zarr datasets
            cy_n = sum([ds.pid.shape[0] for ds in cy_zarrs]) # get total number of elements
        if 'nr' in patches:
            nr_files = sorted(glob.glob(os.path.join(src,'nearest*.zarr'))) # get dataset filenames
            nr_zarrs = [xr.open_zarr(file) for file in nr_files]            # open zarr datasets
            nr_n = sum([ds.pid.shape[0] for ds in nr_zarrs]) # get total number of elements
        if 'rn' in patches:
            rn_files = sorted(glob.glob(os.path.join(src,'random*.zarr')))  # get dataset filenames
            rn_zarrs = [xr.open_zarr(file) for file in rn_files]            # open zarr datasets
            rn_n = sum([ds.pid.shape[0] for ds in rn_zarrs]) # get total number of elements
        # get the total number of elements of the entire dataset
        mul = len(augmentation) + 1 # number of augmentations + baseline data
        self.n = cy_n * mul + nr_n + rn_n
        # save cy_n for augmentation purposes
        self.cy_n = cy_n
        # get dataset from the zarr files
        self._prepare_dataset(cy_zarrs, nr_zarrs, rn_zarrs, drivers, targets)
        # prepare for the scaling
        if self.scaler:
            _, self.C, self.H, self.W = self.x_data['cy'].shape

    def __len__(self):
        return self.n

    def __getitem__(self, index: int) -> Any:
        bucket, idx = index
        # get data from the correct bucket
        key = list(self.x_data.keys())[bucket]
        x, y = self.x_data[key][idx], self.y_data[key][idx]
        # scale the features
        x = self._scale(x)
        # augment data
        x, y = self._augment(x, y, bucket)
        # apply no cyclone label
        y = self._apply_no_cyclone_label(y)
        # cast the tensor to desired dtype
        x, y = x.type(torch.float32), y.type(self.dtype)
        if self.label_no_cyclone is not None:
            # add classification target (plus small epsilon for regularization)
            y_cls = torch.zeros_like(y[:,0]) + torch.rand_like(y[:,0]) * 0.12
            # set to 1.0 all the positive TCs (minus small epsilon for regularization)
            y_cls[torch.where(y[0,0] != self.label_no_cyclone)[0]] = 1.0 - torch.rand_like(y[:,0]) * 0.12
            # return the classification
            if self.classify: return x, y_cls
            # return the coordinates
            else: return x, y[0]
        else:
            # we are producing patches in output
            return x, y * self.mag

    def _apply_no_cyclone_label(self, y: torch.Tensor):
        if self.label_no_cyclone is not None:
            return torch.where(y < 0, self.label_no_cyclone, y)
        else: 
            return y

    def _prepare_dataset(self, 
            cy_zarrs: List[xr.Dataset], 
            nr_zarrs: List[xr.Dataset], 
            rn_zarrs: List[xr.Dataset], 
            drivers: List[str], 
            targets: List[str]):
        self.x_data, self.y_data = {}, {}
        if cy_zarrs is not None:
            # cyclone data
            logging.info(f'reading cyclone data')
            x_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, drivers, self.dtype)
            y_cy_data = read_zarrs_as_torch_tensor(cy_zarrs, targets, self.dtype)
            logging.info(f'  adding cyclone data')
            self.x_data.update({'cy' : x_cy_data})
            self.y_data.update({'cy' : y_cy_data})
            for aug in self.augmentation:
                logging.info(f'  adding augmentation {aug}')
                self.x_data.update({aug: torch.clone(x_cy_data)})
                self.y_data.update({aug: torch.clone(y_cy_data)})
        if nr_zarrs is not None:
            # nearest data
            logging.info(f'reading nearest data')
            x_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, drivers, self.dtype)
            y_nr_data = read_zarrs_as_torch_tensor(nr_zarrs, targets, self.dtype)
            logging.info(f'  adding nearest data')
            self.x_data.update({'nr' : x_nr_data})
            self.y_data.update({'nr' : y_nr_data})
        if rn_zarrs is not None:
            # random data
            logging.info(f'reading random data')
            x_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, drivers, self.dtype)
            y_rn_data = read_zarrs_as_torch_tensor(rn_zarrs, targets, self.dtype)
            logging.info(f'  adding random data')
            self.x_data.update({'rn' : x_rn_data})
            self.y_data.update({'rn' : y_rn_data})
        self.total_num_samples = [x.shape[0] for x in self.x_data.values()]

    def _augment(self, x, y, index):
        for i, aug in enumerate(self.augmentation):
            if aug == 'rot' and index == i + 1:
                if self.label_no_cyclone is not None:
                    x, y = coo_rot180(data=(x,y))
                else:
                    x, y = rot180(data=(x,y))
                return x, y
            elif aug == 'ud' and index == i + 1:
                if self.label_no_cyclone is not None:
                    x, y = coo_up_down(data=(x,y))
                else:
                    x, y = up_down(data=(x,y))
                return x, y
            elif aug == 'lr' and index == i + 1:
                if self.label_no_cyclone is not None:
                    x, y = coo_left_right(data=(x,y))
                else:
                    x, y = left_right(data=(x,y))
                return x, y
        return x, y

    def _scale(self, x: torch.Tensor):
        if self.scaler:
            x = torch.permute(x, dims=(1,2,0)) # C x H x W -> H x W x C
            x = self.scaler.transform(x)
            x = torch.permute(x, dims=(2,0,1)) # H x W x C -> C x H x W
        return x
