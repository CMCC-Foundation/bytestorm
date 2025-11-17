#!/bin/sh
#BSUB -n 2
#BSUB -q g_long
#BSUB -P R000
#BSUB -R "span[ptile=2] rusage[mem=900G]"
#BSUB -J training
#BSUB -o ./job.out
#BSUB -e ./job.err
#BSUB -gpu "num=2"


# run one of the two
mpirun python training.py -c config/config_classification_model.toml --devices 2 --num_nodes 1
mpirun python training.py -c config/config_localization_model.toml --devices 2 --num_nodes 1