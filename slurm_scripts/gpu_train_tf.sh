#!/bin/sh
#SBATCH --job-name=train_tf
#SBATCH --nodes=1
#SBATCH -p gpu_shared
#SBATCH --time=12:00:00
cd ..
python3 main.py model=tf debug=True
