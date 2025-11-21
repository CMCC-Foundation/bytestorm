# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import itertools
import random


def get_nocyclones_patches(dataset, cyclone_patch_ids, patch_size=40):
    """
    Given a dataset and a list of coordinates of patches containing cyclones, 
    we get a list of patches that do not contain a cyclone.
    
    Parameters
    ----------
    dataset:
        It is the xr.Dataset from a timestep containing the map to be divided into patches.
    patch_cyclone_ids:
        It is a list of row-column coordinates of the map patches.

    """
    row_blocks = len(dataset.lat.data) // patch_size
    col_blocks = len(dataset.lon.data) // patch_size

    patch_row_idx = [i for i in range(row_blocks)]
    patch_col_idx = [i for i in range(col_blocks)]

    # get all coordinates
    all_coords = [coords for coords in list(itertools.product(patch_row_idx, patch_col_idx))]

    # remove from all coordinates the TC coords
    no_tc_coords = list(set(all_coords).difference(set(cyclone_patch_ids)))

    nocyclone_patch_ids = set()
    for coo in no_tc_coords:
        nocyclone_patch_ids.add(coo)

    return nocyclone_patch_ids


def get_all_adjacent_patches(dataset, patch_cyclone_ids, patch_size=40):
    """
    Given a list of id patches happy cyclones, you get all the patches in the neighborhood.

    Parameters
    ----------
    dataset:
        It is the xr.Dataset from a timestep containing the map to be divided into patches.
    patch_cyclone_ids:
        It is a list of row-column coordinates of the map patches.

    """
    row_blocks = len(dataset.lat.data) // patch_size
    col_blocks = len(dataset.lon.data) // patch_size
    tmp_all_adjacent_ids = []
    for patch_id in patch_cyclone_ids:
        i, j = patch_id
        tmp_all_adjacent_ids += [a for a in itertools.product([i-1, i, i+1], [j-1, j, j+1])]
    all_adjacent_ids = set()
    for aa_id in tmp_all_adjacent_ids:
        i, j = aa_id
        if 0 <= i < row_blocks and 0 <= j < col_blocks:
            all_adjacent_ids.add(aa_id)
    all_adjacent_ids = all_adjacent_ids.difference(set(list(map(tuple, patch_cyclone_ids))))
    return all_adjacent_ids


def get_nearest_adjacent_patches(dataset, patch_cyclone_ids, patch_cyclone_positions, patch_size=40):
    """
    Given a list of patches ids containing cyclones and their local locations (in the patch), 
    the three patches closest to the cyclone are obtained.

    Parameters
    ----------
    dataset:
        It is the xr.Dataset from a timestep containing the map to be divided into patches.
    patch_cyclone_ids:
        It is a list of row-column coordinates of the map patches.
    patch_cyclone_positions:
        It is a list of row-column coordinates that identify the position of the cyclone within the relevant patch.

    """
    def is_first_quadrant(y, x, half_patch_size):
        return y < half_patch_size and x >= half_patch_size
    def is_second_quadrant(y, x, half_patch_size):
        return y < half_patch_size and x < half_patch_size
    def is_third_quadrant(y, x, half_patch_size):
        return y >= half_patch_size and x < half_patch_size
    def is_fourth_quadrant(y, x, half_patch_size):
        return y >= half_patch_size and x >= half_patch_size

    row_blocks = len(dataset.lat.data) // patch_size
    col_blocks = len(dataset.lon.data) // patch_size
    nearest_patch_ids = set()
    half_patch_size = patch_size / 2
    for patch_id, cyclone_position in zip(patch_cyclone_ids, patch_cyclone_positions):
        i,j = patch_id
        y,x = cyclone_position
        if is_first_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i-1, j   )) if i-1 >= 0 else None
            nearest_patch_ids.add(( i-1, j+1 )) if i-1 >= 0 and j+1 < col_blocks else None
            nearest_patch_ids.add(( i  , j+1 )) if j+1 < col_blocks else None
        elif is_second_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i-1, j-1 )) if i-1 >= 0 and j-1 >= 0 else None
            nearest_patch_ids.add(( i  , j-1 )) if j-1 >= 0 else None
            nearest_patch_ids.add(( i-1, j   )) if i-1 >= 0 else None
        elif is_third_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i  , j-1 )) if j-1 >= 0 else None
            nearest_patch_ids.add(( i+1, j-1 )) if i+1 < row_blocks and j-1 >= 0 else None
            nearest_patch_ids.add(( i+1, j   )) if i+1 < row_blocks else None
        elif is_fourth_quadrant(y, x, half_patch_size):
            nearest_patch_ids.add(( i  , j+1 )) if j+1 < col_blocks else None
            nearest_patch_ids.add(( i+1, j+1 )) if i+1 < row_blocks and j+1 < col_blocks else None
            nearest_patch_ids.add(( i+1, j   )) if i+1 < row_blocks else None
    # remove patch cyclone ids from the nearest ones
    nearest_patch_ids = nearest_patch_ids.difference(set(list(map(tuple, patch_cyclone_ids))))
    return nearest_patch_ids


def get_random_patches(dataset, patch_cyclone_ids, patch_size=40):
    """
    Given a dataset and a list of coordinates of patches containing cyclones, 
    we obtain a list (of length equal to the number of cyclones present in the sample) 
    of random patches that are not located in the neighborhood of the considered patch.

    Parameters
    ----------
    dataset:
        E' il xr.Dataset da un timestep contentente la mappa da suddividere in patch.
    patch_cyclone_ids:
        E' una lista di coordinate riga-colonna delle patch della mappa.

    """
    row_blocks = len(dataset.lat.data) // patch_size
    col_blocks = len(dataset.lon.data) // patch_size

    patch_row_idx = [i for i in range(row_blocks)]
    patch_col_idx = [i for i in range(col_blocks)]

    inv_coords = []
    for patch_id in patch_cyclone_ids:
        i, j = patch_id

        #(5,5)
        #(4,5),(5,4),(4,4),(6,6),(6,5),(5,6),(6,4),(4,6)
        inv_i = [i-1, i, i+1]
        inv_j = [j-1, j, j+1]
        inv_coords.extend(list(itertools.product(inv_i, inv_j)))

    no_tc_coords_candidates = [coords for coords in list(itertools.product(patch_row_idx, patch_col_idx)) if coords not in inv_coords]
    no_tc_coords = random.sample(no_tc_coords_candidates, len(patch_cyclone_ids))

    random_patch_ids = set()
    for coo in no_tc_coords:
        random_patch_ids.add(coo)

    return random_patch_ids
