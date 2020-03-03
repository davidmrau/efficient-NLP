#!/bin/sh
#SBATCH --job-name=train_tf
#SBATCH --nodes=1
#SBATCH -p gpu_shared
#SBATCH --time=12:00:00
cd ..
python3 main.py model=tf tf.pretrained_embeddings=True debug=True  sparse_dimensions=1000
