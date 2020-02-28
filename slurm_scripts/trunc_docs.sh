#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 --ntasks-per-node=1 
#SBATCH --time=00:45:00
python3 trunc_docs.py
