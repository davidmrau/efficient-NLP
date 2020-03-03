#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 
#SBATCH -p gpu_shared
#SBATCH --time=12:00:00
cd ..
python3 main.py snrm.hidden_sizes='200-' sparse_dimensions=10000 l1_scalar=0.1
