module purge
# module load git
module load cmake/3.15.4
module load anaconda3/2023.03
module load nvidia-hpc-sdk/22.2
module load cuda/10.2
module load hdf5/1.12.0
module load gcc/10.1.0-offload
conda activate /gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new
module unload nvidia-hpc-sdk/22.2
module load nvidia-hpc-sdk/22.2
module load patchelf

export CUDNN_LIBRARY=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib
export CUDNN_INCLUDE_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/include
export PYTHONPATH=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages
export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH:/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/torch/lib
export Torch_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/torch/share/cmake/Torch
export CFLAGS="$CFLAGS -I/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow/include"
export CPATH=/apps/GCC/10.1.0-offload
export SMARTSIM_REDISAI=1.2.5
export Tensorflow_BUILD_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow
export SMARTREDIS_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/SmartRedis/install
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
export NO_CHECKS=1
