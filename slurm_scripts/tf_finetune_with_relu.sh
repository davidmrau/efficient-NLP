#!/bin/bash
# Set job requirements
#SBATCH --job-name=tf_fine
#SBATCH --ntasks=1
#SBATCH --partition=gpu_shared
#SBATCH --time=08:00:00
#SBATCH --mem=100G


#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kondilidisn9@gmail.com

#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176


cd ..

EXPERIMENTS_DIR="experiments"

LOAD_MODEL_FOLDER="l1_1_Emb_bert_Sparse_5000_bsz_128_lr_0.0001_TF_L_2_H_4_D_768_P_AVG_ACT_delu"

DATA_PATH="../../data"


python3 finetune_relu.py experiments_dir=${EXPERIMENTS_DIR} data_path=${DATA_PATH} load_model_folder=${LOAD_MODEL_FOLDER} 

LOAD_MODEL_FOLDER="l1_0_Emb_bert_Sparse_5000_bsz_128_lr_0.0001_TF_L_2_H_4_D_768_P_AVG_ACT_delu"

python3 finetune_relu.py experiments_dir=${EXPERIMENTS_DIR} data_path=${DATA_PATH} load_model_folder=${LOAD_MODEL_FOLDER} 
