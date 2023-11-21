#!/bin/bash
#
# SMARTSOD2D virtual environment for python
#
# Bernat Font, Arnau Miro (c) 2023

# Parameters
PREFIX=/apps/python/venv
SMARTSIM_VERS=0.4.2
NUMPY_VERS=1.20.0
SCIPY_VERS=1.6.0
MATPLOTLIB_VERS=3.7.0
TENSORFLOW_VERS=2.8.0
TFAGENTS_VERS=0.10.0
TFPROB_VERS=0.14.1

# Modules to be loaded
module purge
module load gcc python cmake

# Installation workflow
python -m venv smartsod2d
source ./smartsod2d/bin/activate
export CC=gcc CXX=g++ NO_CHECKS=1

pip install smartsim==${SMARTSIM_VERS}

pip install smartredis
pip install scipy==${SCIPY_VERS}
pip install matplotlib==${MATPLOTLIB_VERS}
pip install numpy==${NUMPY_VERS}
pip install tensorflow==${TENSORFLOW_VERS}
pip install tf_agents==${TFAGENTS_VERS}
pip install tensorflow-probability==${TFPROB_VERS}

smart build --device cpu --no_pt --no_tf

# Copy to venv folder
deactivate
sudo mkdir -p $PREFIX
sudo mv smartsod2d $PREFIX
