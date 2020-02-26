#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1
#SBATCH --time=24:00:00
##SBATCH -p gpu_shared
python3 main.py
