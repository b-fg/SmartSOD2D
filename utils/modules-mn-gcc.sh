ml purge && ml load gcc/9.2.0 openmpi/3.1.1 hdf5/1.12.2 python/3.10.2 gmsh cmake/3.15.4
source /gpfs/projects/bsc21/bsc21850/sod_drl/apps/cpu/smartsod2d-gcc/bin/activate

export SMARTREDIS_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/cpu/smartredis/0.4.0/gcc/9.2.0/install
export PATH=$SMARTREDIS_DIR/bin:$PATH
export LD_LIBRARY_PATH=$SMARTREDIS_DIR/lib:$LD_LIBRARY_PATH

export RAI_DIR=/gpfs/projects/bsc21/bsc21850/sod_drl/apps/cpu/rediai/build 
export RAI_PATH=$RAI_DIR/redisai.so
export LD_LIBRARY_PATH=$RAI_DIR/lib:$LD_LIBRARY_PATH
 
