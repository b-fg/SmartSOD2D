#!/bin/bash

cdir=$(pwd)
clean=$1

cd # go to home dir, where smartsod2d does not exist
cd smartsod2d
cd utils

# Load modules if they are not loaded
module list 2> spoolfile # cat spoolfile | grep -wq "nvidia")
if ! cat spoolfile | grep -wq "nvidia"; then
    source modules-p9.sh
fi
rm spoolfile

# Configure
cd smartsod2d
cd sod2d_gitlab

if [ $# -ne 0 ]; then # arguments supplied, clean compilation
    git checkout -- CMakeLists.txt
    # echo "add_compile_options(--gcc-toolchain=/apps/GCC/10.1.0-offload/bin/gcc)" >> CMakeLists.txt
    rm -rf build && mkdir build && cd build
    cmake -DUSE_PCPOWER=ON -DUSE_GPU=ON -DUSE_MEM_MANAGED=ON -DUSE_SMARTREDIS=ON -DTOOL_MESHPART=OFF ..
else
    cd build
fi

# Compile
make -j

# Link correct GCC
patchelf --replace-needed libstdc++.so.6 /apps/GCC/10.1.0-offload/lib/gcc/ppc64le-redhat-linux/10.1.0/libstdc++.so.6 src/app_sod2d/sod2d

# Return to current dir
cd $cdir
