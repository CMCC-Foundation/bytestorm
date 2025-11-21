# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# ByteStorm Library for Tropical Cyclone Tracking           #
# Copyright (c) 2025 CMCC Foundation                        #
# Licensed under The GPLv3 License [see LICENSE for details]#
# Written by Davide Donno                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

import warnings
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib.ticker as mticker
from matplotlib import pyplot as plt
import matplotlib.transforms as transforms
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter)


def plot_tracks(det_tracks, obs_tracks, lat_range, lon_range, title, outfile=None):
    # set map extent
    central_longitude = (lon_range[1] - lon_range[0])

    fig = plt.figure(figsize=(25,10))
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax = plt.axes(projection=proj)

    image_extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    ax.set_extent(image_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', lw=0.2)
    ax.add_feature(cf.LAND, facecolor='lightgrey', alpha=0.3)

    fontdict = {'weight':'bold', 'size':14}
    title_fontdict = {'size':18}
    ticksize = 12
    track_id_col = 'TRACK_ID' if 'TRACK_ID' in det_tracks.columns else 'track_id'
    lon_col = 'LON' if 'LON' in det_tracks.columns else 'lon'
    lat_col = 'LAT' if 'LAT' in det_tracks.columns else 'lat'

    # plot tracks in each basin
    alpha = 0.5
    marker_size = 10.0
    transform = ccrs.PlateCarree()
    plt.title(title, fontsize = 18)
    if obs_tracks is not None:
        for i,id in enumerate(obs_tracks[track_id_col].unique()):
            ax.plot(obs_tracks[obs_tracks[track_id_col]==id][lon_col], obs_tracks[obs_tracks[track_id_col]==id][lat_col], alpha=0.2, transform=transform, color='blue')
            ax.scatter(obs_tracks[obs_tracks[track_id_col]==id][lon_col], obs_tracks[obs_tracks[track_id_col]==id][lat_col], s=marker_size, marker='o', alpha=0.9, transform=transform, color='blue', label=f'Observed Tracks (#{len(obs_tracks[track_id_col].unique())})' if i==0 else None)
    for i,id in enumerate(det_tracks[track_id_col].unique()):
        ax.plot(det_tracks[det_tracks[track_id_col]==id][lon_col], det_tracks[det_tracks[track_id_col]==id][lat_col], alpha=0.2, transform=transform, color='red')
        ax.scatter(det_tracks[det_tracks[track_id_col]==id][lon_col], det_tracks[det_tracks[track_id_col]==id][lat_col], s=marker_size, marker='o', alpha=0.9, transform=transform, color='red', label=f'Detected Tracks (#{len(det_tracks[track_id_col].unique())})' if i==0 else None)
    # x-axis
    longitudes = np.arange(lon_range[0], lon_range[1]+1, 10)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.xaxis.set_major_locator(mticker.FixedLocator(longitudes-central_longitude))
    ax.set_xticklabels(longitudes, size=ticksize)
    ax.set_xticks(longitudes-central_longitude)
    ax.set_xlabel('Longitude [deg]', fontdict=fontdict)

    # y-axis
    latitudes = np.arange(lat_range[0], lat_range[1]+1, 10)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_major_locator(mticker.FixedLocator(latitudes))
    ax.set_yticklabels(latitudes, size=ticksize)
    ax.set_yticks(latitudes)
    ax.set_ylabel('Latitude [deg]', fontdict=fontdict)

    # ax.set_title(f"TSTORMS and ML inference on CMCC-CM3 (#{len(df)} matches)", fontdict=title_fontdict)

    # gridlines
    gl = ax.gridlines(crs=proj, draw_labels=False, dms=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(longitudes-central_longitude)
    gl.ylocator = mticker.FixedLocator(latitudes)
    gl.xlines = True
    gl.ylines = True

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='upper left', markerscale=2., edgecolor='black', framealpha=1, ncol=4, fontsize=14, bbox_to_anchor=(0.29,-0.11))

    if outfile: 
        plt.savefig(f'{outfile}', dpi=300, bbox_inches='tight')
    plt.show()

def plot_storm_track_density(transits, algo, vmin, vmax, remove_atlantic, lat_range, lon_range):
    if remove_atlantic:
        lon_grid, lat_grid = np.meshgrid(np.linspace(100, 270, transits.shape[1]), np.linspace(*lat_range, transits.shape[0])[::-1])
    else:
        lon_grid, lat_grid = np.meshgrid(np.linspace(*lon_range, transits.shape[1]), np.linspace(*lat_range, transits.shape[0])[::-1])

    # set map extent
    central_longitude = (lon_range[1] - lon_range[0])

    _ = plt.figure(figsize=(16,4))
    proj = ccrs.PlateCarree(central_longitude=central_longitude)
    ax = plt.axes(projection=proj)

    image_extent = [lon_range[0], lon_range[1], lat_range[0], lat_range[1]]
    ax.set_extent(image_extent, crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m', lw=0.2)
    ax.add_feature(cf.LAND, facecolor='lightgrey', alpha=0.3)

    fontdict = {'weight':'bold', 'size':14}
    title_fontdict = {'size':18}
    ticksize = 12

    im = ax.pcolormesh(lon_grid, lat_grid, transits, cmap='terrain', transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax)

    # x-axis
    longitudes = np.arange(lon_range[0], lon_range[1]+1, 40)
    lon_formatter = LongitudeFormatter(zero_direction_label=False)
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.xaxis.set_major_locator(mticker.FixedLocator(longitudes-central_longitude))
    ax.set_xticklabels(longitudes, size=ticksize)
    ax.set_xticks(longitudes-central_longitude)
    ax.set_xlabel('Longitude [deg]', fontdict=fontdict)

    # y-axis
    latitudes = np.arange(lat_range[0], lat_range[1]+1, 20)
    lat_formatter = LatitudeFormatter()
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.yaxis.set_major_locator(mticker.FixedLocator(latitudes))
    ax.set_yticklabels(latitudes, size=ticksize)
    ax.set_yticks(latitudes)
    ax.set_ylabel('Latitude [deg]', fontdict=fontdict)

    ax.set_title(f"Tropical Storm Track Density ({algo.upper()})", fontdict=title_fontdict)

    # gridlines
    gl = ax.gridlines(crs=proj, draw_labels=False, dms=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator(longitudes-central_longitude)
    gl.ylocator = mticker.FixedLocator(latitudes)
    gl.xlines = True
    gl.ylines = True

    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    # plt.show()

def plot_pod_and_far(algo_results: pd.DataFrame, label, outfile = None):
    x = np.arange(len(algo_results))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0
    
    fig, ax = plt.subplots(layout='constrained', figsize=(12,6))
    
    # add horizontal mean
    ax.hlines(y=algo_results['pod'].mean(), xmin=-0.3, xmax=4.65, zorder=0, color='tab:blue', label='Average POD')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, algo_results['pod'].mean(), "{:.0f} %".format(algo_results['pod'].mean()), color="tab:blue", transform=trans, ha="right", va="center")
    
    ax.hlines(y=algo_results['far'].mean(), xmin=-0.3, xmax=4.65, zorder=999, color='tab:red', label='Average FAR')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, algo_results['far'].mean(), "{:.0f} %".format(algo_results['far'].mean()), color="tab:red", transform=trans, ha="right", va="center")
    
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(algo_results['pod'],2), width, label=f'POD' + label, color='tab:blue')
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(algo_results['far'],2), width, label=f'FAR' + label, color='tab:red')
    ax.bar_label(rects, padding=3)
    multiplier += 1
    
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score', size = 14)
    ax.set_xlabel('TC trackers', size = 14)
    ax.set_xticks(x + width/2, algo_results['algo'], fontsize=12)
    ax.set_ylim(0, 100)
    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper left', markerscale=10, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(0.2 , 0.98))
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()

def plot_pod_and_far_multi_trackers(algo_results: pd.DataFrame, trackers: dict, label: str, rotation: float = 90, outfile: str = None):
    x = np.arange(len(algo_results))  # the label locations
    width = 0.4  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained', figsize=(2*len(trackers.keys()),6))

    # add horizontal mean
    ax.hlines(y=algo_results['pod'].mean(), xmin=-0.3, xmax=len(trackers.keys())-1.2, zorder=0, color='tab:blue', label='Average POD')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, algo_results['pod'].mean(), "{:.0f} %".format(algo_results['pod'].mean()), color="tab:blue", transform=trans, ha="right", va="center")

    ax.hlines(y=algo_results['far'].mean(), xmin=-0.3, xmax=len(trackers.keys())-1.2, zorder=0, color='tab:red', label='Average FAR')
    trans = transforms.blended_transform_factory(ax.get_yticklabels()[0].get_transform(), ax.transData)
    ax.text(0, algo_results['far'].mean(), "{:.0f} %".format(algo_results['far'].mean()), color="tab:red", transform=trans, ha="right", va="center")

    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(algo_results['pod'],2), width, label=f'POD '+label, color='tab:blue')
    ax.bar_label(rects, padding=3)
    multiplier += 1

    offset = width * multiplier
    rects = ax.bar(x + offset, np.round(algo_results['far'],2), width, label=f'FAR '+label, color='tab:red')
    ax.bar_label(rects, padding=3)
    multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Score')
    ax.set_xlabel('TC trackers')
    ax.set_xticks(x + width/2, algo_results['algo'], rotation=rotation, fontsize=12)
    ax.set_ylim(0, 100)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    ax.legend(loc='upper left', markerscale=8, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(0.2 , 1.13))

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()

def plot_trackers_biases_wrt_baseline(baseline, biases, tracker_names, comp1, comp2, ylims, ytwlims, outfile=None):
    nrows = len(tracker_names)
    ncols = len(biases.keys())
    _, axes = plt.subplots(nrows=nrows, ncols=1, layout='constrained', figsize=(1 * nrows, 6 * ncols))
    baseline1 = baseline[f'{comp1}']
    baseline2 = baseline[f'{comp2}']
    plt.suptitle(f'{comp1.replace("_", " ").upper()} and {comp2.replace("_", " ").upper()} Bias w.r.t. Baseline Model ({comp1.replace("_", " ").upper()} : {baseline1}, {comp2.replace("_", " ").upper()} : {baseline2})')
    for j, (ax, tname) in enumerate(zip(axes, tracker_names)):
        ax.hlines(y=0, xmin=-1000, xmax=1000, zorder=0, color='black', linewidth=1, linestyles='dashed')
        width, multiplier, rotation = 0.4, 0, 0
        x = np.arange(ncols)
        axtw = ax.twinx()
        for i,(key, value) in enumerate(biases.items()):
            offset = width * multiplier
            rects1 = ax.bar(x[i:i+1] + offset, value[value['tracker']==tname][f'{comp1}-bias'], width, label=f'{comp1.upper()} bias', color='tab:blue')
            ax.bar_label(rects1, padding=2)
            
            rects2 = axtw.bar(x[i:i+1] + offset, value[value['tracker']==tname][f'{comp1}'], width, label=f'{comp1.upper()}', edgecolor='tab:blue', color='none', hatch="/")
            axtw.bar_label(rects2, padding=2)
            
            multiplier += 1
            
            offset = width * multiplier
            
            rects3 = ax.bar(x[i:i+1] + offset, value[value['tracker']==tname][f'{comp2}-bias'], width, label=f'{comp2.upper()} bias', color='tab:red')
            ax.bar_label(rects3, padding=2)
            
            rects4 = axtw.bar(x[i:i+1] + offset, value[value['tracker']==tname][f'{comp2}'], width, label=f'{comp2.upper()}', edgecolor='tab:red', color='none', hatch="/")
            axtw.bar_label(rects4, padding=2)
            
            multiplier += 1
            
            bbox = dict(boxstyle='round', edgecolor='green', facecolor='white', alpha=1.0)
            ax.vlines(x=x[i] + offset + width + 0.3, ymin=ylims[0], ymax=ylims[1], color='green', linestyle='dashed')
            ax.text(x[i:i+1] + offset - width * 2, 0, key.upper().replace('_', ' '), rotation=90, fontsize=8, color='green', horizontalalignment='center', verticalalignment='center', bbox=bbox)
        
        if j == len(axes)-1:
            lns = [rects1, rects2, rects3, rects4]
            labs = [l.get_label() for l in lns]
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height])
            ax.legend(lns, labs, loc='upper left', markerscale=8, edgecolor='gray', framealpha=1, ncol=4, bbox_to_anchor=(0.15 ,-0.04))
        
        ax.set_ylabel(f'Bias\n{tname}')
        ax.set_xticks([])
        ax.set_ylim(*ylims)
        ax.set_xlim(-1, 10)
        axtw.set_ylim(*ytwlims)
        
        if ylims[0] / ylims[1] < ytwlims[0] / ytwlims[1]:
            axtw.set_ylim(bottom = ytwlims[1] * (ylims[0] / ylims[1]))
        else:
            ax.set_ylim(bottom = ylims[1] * (ytwlims[0] / ytwlims[1]))
        
    if outfile:
        plt.savefig(outfile, dpi=300)
    else:
        plt.show()
