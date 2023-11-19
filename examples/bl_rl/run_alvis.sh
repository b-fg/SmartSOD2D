#!/bin/bash

#SBATCH --job-name=marl-8e
#SBATCH --account=NAISS2023-5-102
#SBATCH --mail-user=bernat.font@bsc.es
#SBATCH --mail-type=all
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH -C NOGPU -n 1
#SBATCH --time=168:00:00

. /mimer/NOBACKUP/groups/deepmechalvis/bernat/smartsod2d/utils/modules-alvis-python.sh
rm -rf sod2d_exp
python run.py