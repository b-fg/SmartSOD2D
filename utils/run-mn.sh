#!/bin/bash

cdir=$(pwd)
python_file=$1
args="${@:2}"

# Load modules if necessary
cd smartsod2d
cd utils
module list 2> spoolfile # cat spoolfile | grep -wq "nvidia")
if ! cat spoolfile | grep -wq "openmpi"; then
    source modules-mn.sh
fi
rm spoolfile

# Copy sod2d executable
cd smartsod2d
cd ../apps/cpu/sod2d_gitlab/build/src/app_sod2d/
cp sod2d $cdir

# Return to working directory
cd $cdir
python $args $python_file
