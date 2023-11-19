#!/bin/bash

# Load modules
. modules-p9.sh

# Add into CMakeLists.txt: 
option(SR_PYTHON  "Build the python module" ON)
option(SR_FORTRAN "Build the fortran client library" ON)
add_compile_options(--gcc-toolchain=/apps/GCC/10.1.0-offload)

# cmake with
cmake .. -DCMAKE_CXX_COMPILER=nvc++ -DCMAKE_C_COMPILER=nvc -DCMAKE_Fortran_COMPILER=nvfortran
