#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 
#SBATCH -p gpu_shared
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
python3 main.py