# bytestorm
This is the official repository for the paper entitled: "ByteStorm: a multi-step data driven approach for Tropical Cyclones detection and tracking"


Fully Data-Driven Tropical Cyclone Tracking
===========================================

## Overview
This repository includes the code for the development and test of a fully data-driven model for Tropical Cyclone (TC) Tracking.

The main goal is to make use of Pangu Weather model in a different setup that could successfully localize and track TC centers.

## The Backbone Architecture
The architecture of our ML model is based on the pseudocode defined defined in the <a href="https://github.com/198808xc/Pangu-Weather">Pangu</a> model. However, the dataset setup is slightly different from the conceptual Pangu model.

Contributors
============

Advanced Scientific Computing (ASC) Division 

* Davide Donno (davide.donno@cmcc.it)

Content
=======
- [Tracking](#tracking)
    - [Case Study](#case-study)

Tracking
========

Case study
----------
The TCTracker is focused in all the world basins and trained using Copernicus ERA5 data retrieved from WeatherBench2. Reanalysis data is 6-hourly on a regular grid with a spatial resolution of $0.25^{\circ}$ ($\sim$ $27$ km). The data shape is made of $721 \times 1440$ points. The model setup is structured as follows:

* *2D input* : consists of 3 variables defining TC centers in different ways:
    * *Density Map* &rarr; map of TC density built using gaussian filter with $\sigma = 10$ representing probability of having a TC.
    * *Square Map* &rarr; map containing a square of radius 10 centered on each TC center.
    * *Label Map* &rarr; map containing a square of radius 10 centered on each TC center where each TC occurrence within a time series has a unique label value. This helps in retrieving the TC along a track.
* *3D input* : contains both single levels and pressure levels variables stacked together in a Tensor having shape (721,1440,1,C) where C is the number of variables used.


The MedFormer is tailored on the Mediterranean area and it is trained using the Mediterranean Sea Physics Reanalysis data from $1987$ to $2022$ available on Copernicus service. Forecast is meant as daily mean values over a spatial domain with resolution of $1/24^{\circ}$ ($\sim$ $5$ km). The data shape is made of $1307 \times 380$ points and $18$ vertical levels. The model is trained using as input volumetric fields the temperature, salinity and the currents speed along two directions (*u* and *v*); as surface fields we use the sea surface height, the water flux and the heat flux, exchanged with the atmosphere, and the atmosphere wind stress applied to the ocean surface. The MedFormer outputs the forecast of the prognostic fields temperature, salinity, currents speed and sea surface height at day $t+h$, where $h=1,2,4,7$. The model is then used and evaluated in autoregression mode to produce a forecast for $7$ days ahead. The training dataset includes $30$ years of data from $1987$ to $2016$; $3$ years ($2017-2019$) are used for validation and $($2020-2022$)$ is used for testing. 

### Setup

![Alt text](library/imgs/medformer.png "MedFormer Setup")

## Data

### Overview

The following Table provides an overview of the variables used to train the MedFormer architecture.

| # | Type | Short Name | Standard Name | Shape | Units |
| :---: | :---: | :---: | :---: | :---: | :---: | 
| 1 | 2D | sossheig | sea_surface_height_above_geoid | (**time**, lat, lon) | $m$ |
| 2 | 2D | sozotaux | surface_downward_x_stress | (**time**, lat, lon) | $\frac{N}{m^2}$ |
| 3 | 2D | sometauy | surface_downward_y_stress | (**time**, lat, lon) | $\frac{N}{m^2}$ |
| 4 | 2D | sowaflup | water_flux_out_of_sea_ice_and_sea_water | (**time**, lat, lon) | $\frac{kg}{\frac{m^2}{s}}$ |
| 5 | 2D | sohefldo | surface_downward_heat_flux_in_sea_water | (**time**, lat, lon) | $\frac{W}{m^2}$ |
| 6 | 3D | vosaline | sea_water_practical_salinity | (**time**, depth, lat, lon) | $1e^{-3}$ |
| 7 | 3D | votemper | sea_water_potential_temperature | (**time**, depth, lat, lon) | $^{\circ}C$ |
| 8 | 3D | vozocrtx | sea_water_x_velocity | (**time**, depth, lat, lon) | $\frac{m}{s}$ |
| 9 | 3D | vomecrty | sea_water_y_velocity | (**time**, depth, lat, lon) | $\frac{m}{s}$ |

Note: **time** is an implicit dimension.

### Input Data Preparation

Both 2D and 3D variables were originally provided in 3 NetCDF files (T, U and V), according to a different underlying spatial grid. Please, refer to the template `.nc` files located in the `templates` folder. 

2D input tensors are shaped (5, 380, 1307) where the first dimension represents the 5 surface variables (*sossheig*, *sozotaux*, *sometauy*, *sowaflup*, *sohefldo* **in the exact order**).

3D input tensors are shaped (4, 18, 380, 1307) where the first dimension represents the 4 surface variables (*vosaline*, *votemper*, *vozocrtx*, *vomecrty* **in the exact order**) and the second dimension represents the 18 depth levels (1.02 m, 3.17 m, 5.46 m, 7.92 m, 10.54 m, 19.4 m, 29.89 m, 51.38 m, 72.62 m, 97.93 m, 153.43 m, 203.17 m, 249.92 m, 303.56 m, 398.54 m, 556.41 m, 756.2 m, 971.08 m **in the exact order**).

In both cases, the dimensions of 380 and 1307 represent the size along the latitude and longitude, where the numerical range is [0,379] points and [0,1306] points on a $1/24^{\circ}$ ($\sim$ $5$ km) of spatial resolution. 

**Data can be gathered from the Copernicus marine service (#TODO)**

Conda Environment
-----------------
Python version 3.11.2 or higher is needed.

A conda env containing all the packages and versions required to run the scripts can be created by running the following command:

      conda env create --file medformer_env.yaml

This makes the installation easy and fast. The PyTorch-cuda v11.8 was used to train and test the MedFormer architecture on the <a href="https://www.cmcc.it/super-computing-center-scc">Juno</a> supercomputer at CMCC.

Library Structure
-----------------

The library is structured as follows:
``` bash
├── README.md
├── data
├── experiments
├── library
│   ├── dataloader.py
│   ├── layers.py
│   ├── losses.py
│   ├── macros.py
│   ├── med.py
│   ├── models.py
│   ├── params.py
│   ├── save.py
│   ├── scaling.py
│   ├── utils.py
├── src
│   ├── config
│   │   ├── debug.toml
│   │   ├── inference.toml
│   │   ├── med_train_vX.toml
│   ├── 00_debug.py
│   ├── 01_torch_scaling.py
│   ├── 02_train.py
│   ├── 03_inference.py
│   ├── 04_inference_no_fluxes.py
│   ├── debug.sh
│   ├── inference_no_flux.sh
│   ├── inference.sh
│   ├── launch.sh
│   ├── scaler.sh
```

`data` is a directory that is used to store important files (e.g., scalers, symbolic links to the dataset, etc). 

`experiments` directory is created at runtime and is used to store, in a hierarchical way, the outputs of the training executions.

`library` directory is used to store, in an organized way, all the code that provides support to the `src` scripts during workflow executions. Model and custom losses implementation and training utility functions can be found here.

`src` directory contains the main scripts used to manage the workflow of the repository. Scalers computation as well as training and inference implementations are stored here. This folder must contain all the code that will be executed. 

Note: both `data` and `experiments` directories are **not** explicitly included in the repository as they contain too heavy files that cannot be stored.

### Execute the code

In order to execute the code, the working directory **must be set** to `src` and a `.sh` file must be prepared with the following structure:

**STOP HERE, INTERNAL INFO NOT RELEVANT (EVALUATE)**

```
#!/bin/sh
#BSUB -n n_jobs
#BSUB -q queue_name
#BSUB -P project_code
#BSUB -R "span[ptile=n_jobs_per_node]"
#BSUB -J exec_name
#BSUB -o ./out_file.out
#BSUB -e ./err_file.err
#BSUB -gpu "num=n_gpus_per_node"

python exec_file.py --arguments args
```

In order to apply a job to the Juno cluster, the following command must be executed:

```
bsub < file.sh
```
