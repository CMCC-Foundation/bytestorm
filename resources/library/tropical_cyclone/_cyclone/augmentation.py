# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import torch

def coo_rot180(data):
    X, y = data
    y = y[0]
    patch_size = X.shape[1]
    X = torch.permute(torch.rot90(torch.permute(X, dims=(1,2,0)), k=2, dims=(0,1)), dims=(2,0,1))
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [-y[0] + patch_size -1, -y[1] + patch_size -1]
    return (X, torch.as_tensor(y1).unsqueeze(0))

def coo_left_right(data):
    X,y = data
    y = y[0]
    patch_size = X.shape[1]
    X = torch.permute(torch.fliplr(torch.permute(X, dims=(1,2,0))), dims=(2,0,1))
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [y[0], - y[1] + patch_size -1]
    return (X, torch.as_tensor(y1).unsqueeze(0))

def coo_up_down(data):
    X,y = data
    y = y[0]
    patch_size = X.shape[1]
    X = torch.permute(torch.flipud(torch.permute(X, dims=(1,2,0))), dims=(2,0,1))
    y1 = [-1., -1.]
    if y[0] != -1:
        y1 = [- y[0] + patch_size -1, y[1]]
    return (X, torch.as_tensor(y1).unsqueeze(0))

def rot180(data):
    X, y = data
    X = torch.permute(torch.rot90(torch.permute(X, dims=(1,2,0)), k=2, dims=(0,1)), dims=(2,0,1))
    y = torch.permute(torch.rot90(torch.permute(y, dims=(1,2,0)), k=2, dims=(0,1)), dims=(2,0,1))
    return (X, y)

def left_right(data):
    X,y = data
    X = torch.permute(torch.fliplr(torch.permute(X, dims=(1,2,0))), dims=(2,0,1))
    y = torch.permute(torch.fliplr(torch.permute(y, dims=(1,2,0))), dims=(2,0,1))
    return (X, y)

def up_down(data):
    X,y = data
    X = torch.permute(torch.flipud(torch.permute(X, dims=(1,2,0))), dims=(2,0,1))
    y = torch.permute(torch.flipud(torch.permute(y, dims=(1,2,0))), dims=(2,0,1))
    return (X, y)