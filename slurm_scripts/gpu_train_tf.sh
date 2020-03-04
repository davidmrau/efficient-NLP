#!/bin/sh
#SBATCH --job-name=train_tf
#SBATCH --nodes=1
#SBATCH -p gpu_shared
#SBATCH --time=12:00:00
cd ..

module purge
module load pre2019
module load eb

module load Python/3.6.3-foss-2017b

module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATHs



python3 main.py model=tf tf.pretrained_embeddings=True debug=True  sparse_dimensions=1000
