#!/bin/bash

cdir=$(pwd)
procs=$1
args="${@:2}"

cd smartsod2d
cp sod2d_gitlab/build/src/app_sod2d/sod2d $cdir
cd $cdir
mpirun -np $procs ./sod2d $args
