#!/bin/bash

module purge
conda deactivate

module load nvhpc-hpcx/23.3
module load hdf5/1.12.2
module load smartredis/0.4.0-nvhpc
conda activate sod_smartsim
