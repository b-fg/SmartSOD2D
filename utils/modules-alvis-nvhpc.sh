# Modules necessary to compile/run Sod2D and SmartRedis

ml purge
ml GCC/11.3.0 CMake/3.23.1-GCCcore-11.3.0 OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0 HDF5/1.12.2-nvompi-2022.07 Python/3.10.4-GCCcore-11.3.0

export HDF5_DIR=/apps/Arch/software/HDF5/1.12.2-nvompi-2022.07
export HDF5_HOME=/apps/Arch/software/HDF5/1.12.2-nvompi-2022.07
export SMARTREDIS_DIR=/mimer/NOBACKUP/groups/deepmechalvis/bernat/apps/smartredis/0.4.0/nvhpc/22.7/install/
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
export SMARTSIM_REDISAI=1.2.5
export RAI_PATH=/mimer/NOBACKUP/groups/deepmechalvis/bernat/apps/rediais/1.2.5/install-cpu/redisai.so
