#!/bin/bash
ml purge
ml load nvidia-hpc-sdk/24.3 hdf5/1.14.1-2-nvidia-nvhpcx cmake mkl python/3.11.5-gcc git-lfs gcc/13.2.0-nvidia-hpc-sdk

export SMARTSOD2D_ENV=/gpfs/scratch/bsc21/bsc021850/smartsod2d-env
export SMARTREDIS_DIR=/gpfs/scratch/bsc21/bsc021850/smartredis/install
export RAI_PATH=/gpfs/scratch/bsc21/bsc021850/redisai/install-cpu/redisai.so
export SMARTSIM_REDISAI=1.2.7

export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

unset PYTHONPATH
. $SMARTSOD2D_ENV/bin/activate

# export Tensorflow_BUILD_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow
# export CUDNN_LIBRARY=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib
# export CUDNN_INCLUDE_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/include
# export PYTHONPATH=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages
# export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH:/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/torch/lib
# export Torch_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/torch/share/cmake/Torch
# export CFLAGS="$CFLAGS -I/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow/include"
# export CPATH=/apps/GCC/10.1.0-offload
# export Tensorflow_BUILD_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow
# export NO_CHECKS=1
