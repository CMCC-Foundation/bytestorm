# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The MIT License [see LICENSE for details]  #
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np

def get_global_rowcol_coords(lats, lons, latlon):
    """
    Get the row-col coordinates (considering the map as a matrix) corresponding to the 
    passed latlon geographical coordinates.

    """
    return np.array([[np.argwhere(lats==l)[0][0] for l in latlon[:,0]],[np.argwhere(lons==l)[0][0] for l in latlon[:,1]]]).transpose()

def from_local_to_global(patch_id, patch_row_col, patch_size=40):
    """
    Returns the global row-col coordinates corresponding to the local patch row-col coordinates.

    Parameters
    ----------
    patch_id : tuple(int, int)
        Row-column coordinates of the position of the patch
    patch_row_col : tuple(int, int)
        Row-column coordinates of the TC inside the patch
    patch_size : int
        Size of the patch
    
    Returns
    -------
    global_row_col: tuple(int, int)
        Row-column coordinates of the TC inside the entire domain

    """
    global_row = patch_size * patch_id[0] + patch_row_col[0]
    global_col = patch_size * patch_id[1] + patch_row_col[1]
    return (int(global_row), int(global_col))
