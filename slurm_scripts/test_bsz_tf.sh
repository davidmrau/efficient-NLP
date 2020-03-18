#!/bin/bash
# Set job requirements
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --partition=gpu_short
#SBATCH --time=00:05:00

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kondilidisn9@gmail.com

#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176


MODEL=tf

BATCH_SIZE="400"

cd ..

# test what is the max batch size that can fit in the gpu. If the folloing trains at least one epoch, we are good for the rest experiments that follow
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=8 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=CLS  \
model_folder=Test_Batch_size_exp num_epochs=2