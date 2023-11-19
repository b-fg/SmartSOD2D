#!/bin/bash

#SBATCH --job-name=marl-8e
#SBATCH --mail-user=bernat.font@bsc.es
#SBATCH --mail-type=all
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

. /gpfs/projects/bsc21/bsc21850/sod_drl/smartsod2d/utils/modules-p9.sh
rm -rf sod2d_exp
python run.py
