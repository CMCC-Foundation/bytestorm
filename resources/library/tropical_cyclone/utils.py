# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np

def round_to_grid(x, grid_res=0.25):
    return grid_res * np.round(x/grid_res)
