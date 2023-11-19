# Modules necessary to run SmartSod2D 

ml purge
ml load SciPy-bundle/2020.11-fosscuda-2020b Szip/2.1.1-GCCcore-10.2.0
source /mimer/NOBACKUP/groups/deepmechalvis/bernat/pyenvs/smartsod2d/bin/activate

export SMARTREDIS_DIR=/mimer/NOBACKUP/groups/deepmechalvis/bernat/apps/smartredis/0.4.0/nvhpc/22.7/install
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH
export SMARTSIM_REDISAI=1.2.5
export RAI_PATH=/mimer/NOBACKUP/groups/deepmechalvis/bernat/apps/rediais/1.2.5/install-cpu/redisai.so

# export NVHPC_DIR=/apps/Arch/software/OpenMPI/4.1.4-NVHPC-22.7-CUDA-11.7.0
# export HDF5_DIR=/apps/Arch/software/HDF5/1.12.2-nvompi-2022.07
# export LD_LIBRARY_PATH=$HDF5_DIR/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$NVHPC_DIR/lib:$LD_LIBRARY_PATH
# export PATH=$NVHPC_DIR/bin:$PATH
# export LD_LIBRARY_PATH=$NVHPC_DIR/Linux_x86_64/22.7/compilers/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/apps/Arch/software/GCCcore/11.3.0/lib:$LD_LIBRARY_PATH
