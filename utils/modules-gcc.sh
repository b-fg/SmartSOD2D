#!/bin/bash

module purge
conda deactivate

module load gcc/9.4.0
module load openmpi/4.1.4
module load hdf5/1.12.2
module load smartredis/0.4.0-gcc
conda activate sod_smartsim
