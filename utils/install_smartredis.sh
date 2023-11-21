#!/bin/bash
#
# SMARTREDIS installation for SMART-SOD2D
#
# Bernat Font, Arnau Miro (c) 2023

# Parameters
VERS=0.4.0
BRANCH="f2003"
FOLDER="smartredis-${VERS}-${BRANCH}"
PREFIX="/apps/smartredis/${VERS}"
URL="https://github.com/ashao/SmartRedis"

# Modules to be loaded (depend on the machine)
module purge
module load gcc nvhpc cmake python


## Installation workflow
# Clone github repo
git clone ${URL} --depth=1 --branch ${BRANCH} ${FOLDER}
cd $FOLDER
make clobber
make deps
mkdir build && cd build
cmake .. \
	-DSR_PYTHON=ON \
	-DSR_FORTRAN=ON \
	-DCMAKE_CXX_COMPILER=nvc++ \
	-DCMAKE_C_COMPILER=nvc \
	-DCMAKE_Fortran_COMPILER=nvfortran
make -j $(getconf _NPROCESSORS_ONLN)
make install
# Install
cd ..
sudo mkdir -p ${PREFIX}
sudo cp -r install/* $PREFIX/
