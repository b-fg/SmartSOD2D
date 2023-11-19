#!/bin/bash

cdir=$(pwd)
clean=$1

cd
cd smartsod2d
cd utils

# Load modules if they are not loaded
module list 2> spoolfile # cat spoolfile | grep -wq "nvidia")
if ! cat spoolfile | grep -wq "openmpi"; then
    source modules-mn.sh
fi
rm spoolfile

# Configure
cd smartsod2d
cd ../apps/cpu/sod2d_gitlab

if [ $# -ne 0 ]; then # arguments supplied, clean compilation
    git checkout -- CMakeLists.txt
    rm -rf build && mkdir build && cd build
    cmake -DUSE_SMARTREDIS=ON -DTOOL_MESHPART=OFF ..
else
    cd build
fi

# Compile
make -j

# Return to current dir
cd $cdir
