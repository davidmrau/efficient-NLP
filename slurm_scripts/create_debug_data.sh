#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=01:00:00
cd ..
python3 create_debug_data.py
