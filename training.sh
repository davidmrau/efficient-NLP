#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
python3 main.py
