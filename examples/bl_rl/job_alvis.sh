#!/bin/bash

#SBATCH --job-name=marl8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -C NOGPU -n 1
#SBATCH --time=168:00:00

. modules-alvis-python.sh
python run.py