
#!/bin/bash
# Set job requirements
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --partition=gpu_short


#Loading modules
module purge
module load pre2019
module load eb

module load python/3.5.0
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/hpc/eb/Debian9/cuDNN/7.1-CUDA-8.0.44-GCCcore-5.4.0/lib64:$LD_LIBRARY_PATHs


MODEL=tf

BATCH_SIZE="512"

cd ..

# test what is the max batch size that can fit in the gpu. If the folloing trains at least one epoch, we are good for the rest experiments that follow
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=8 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=CLS  \
model_folder=Test_Batch_size_exp num_epochs=2
