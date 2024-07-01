#!/bin/bash

#SBATCH --job-name=smartsod2d
#SBATCH --account=bsc21
#SBATCH --qos=acc_bsccase
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

. ../../utils/modules-mn5-acc.sh
python run.py