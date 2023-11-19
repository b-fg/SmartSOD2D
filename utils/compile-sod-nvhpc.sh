#!/bin/bash

module purge
conda deactivate

module load nvhpc-hpcx/23.3
module load hdf5/1.12.2
module load smartredis/0.4.0-nvhpc
conda activate sod_smartsim

clean=$1
cdir=$(pwd)

cd smartsod2d/sod2d_gitlab # relexi_sod is in $CPATH
cp ../utils/CMakeLists-nvhpc.txt CMakeLists.txt
cp ../utils/compilerOps-cc75.cmake cmake/compilerOps.cmake

if [ $# -ne 0 ]; then # arguments supplied, clean compilation
    rm -rf build && mkdir build && cd build
    cmake ..
else
    cd build
fi

make -j

cd $cdir # return to current dir
