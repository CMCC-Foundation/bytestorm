# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import numpy as np

def round_to_grid(x, grid_res=0.25):
    return grid_res * np.round(x/grid_res)

def build_rolling_months(start_year, end_year):
    years = range(start_year, end_year)
    year_and_month = []
    k = 0
    for year in years:
        while True:
            year_and_month += [(year, k+1)]
            k += 5
            if k % 12 < k:
                k = k % 12
                break
            k = k % 12
    years_dict = dict()
    for y,m in year_and_month:
        if y not in years_dict.keys():
            years_dict[y] = [i for i in range(13)]
        years_dict[y].remove(m)
    return year_and_month, years_dict