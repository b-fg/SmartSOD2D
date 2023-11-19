ml purge 
ml load GCC/11.2.0 CMake/3.22.1-GCCcore-11.2.0 OpenMPI/4.1.1-GCC-11.2.0 HDF5/1.12.1-gompi-2021b Python/3.9.6-GCCcore-11.2.0 SciPy-bundle/2021.10-foss-2021b h5py/3.6.0-foss-2021b

export HDF5_DIR=/cephyr/users/bernatf/Alvis/apps/hdf5/1.12.1/gcc/11.2
export HDF5_HOME=/cephyr/users/bernatf/Alvis/apps/hdf5/1.12.1/gcc/11.2
export PATH=/cephyr/users/bernatf/Alvis/apps/hdf5/1.12.1/gcc/11.2/bin:$PATH
export LD_LIBRARY_PATH=/cephyr/users/bernatf/Alvis/apps/hdf5/1.12.1/gcc/11.2/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/cephyr/users/bernatf/Alvis/apps/hdf5/1.12.1/gcc/11.2/lib:$IBRARY_PATH

#export CUDNN_LIBRARY=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib
#export CUDNN_INCLUDE_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/include
#export PYTHONPATH=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages
#export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH:/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/torch/lib
#export Torch_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/torch/share/cmake/Torch
#export CFLAGS="$CFLAGS -I/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow/include"
#export CPATH=/apps/GCC/10.1.0-offload
#export SMARTSIM_REDISAI=1.2.5
#export Tensorflow_BUILD_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/lib/python3.9/site-packages/tensorflow
#export SMARTREDIS_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/gpu/smartsim-0.4.2_new/SmartRedis/install
#export PATH=$SMARTREDIS_DIR/bin:$PATH
#export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
#export NO_CHECKS=1
