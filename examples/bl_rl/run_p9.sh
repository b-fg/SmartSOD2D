#!/bin/bash

#SBATCH --job-name=marl8
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00

. modules-p9.sh
python run.py
