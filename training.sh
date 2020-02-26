#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=24:00:00
python3 main.py
