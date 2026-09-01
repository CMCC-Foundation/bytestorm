#!/bin/sh

# mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py SI
# mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py SP
# mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py WP
mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py EP
# mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py NA
# mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py NI
# mpirun -np NPROCS python src/detection/patch/build_dataset_patches_global.py SA