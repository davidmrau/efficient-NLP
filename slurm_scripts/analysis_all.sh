#!/bin/bash
# Set job requirements
#SBATCH --job-name=anal
#SBATCH --ntasks=1
#SBATCH --partition=gpu_shared
#SBATCH --time=24:00:00
#SBATCH --mem=100G

BATCH_SIZE="128"

#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176


QRELS_PATH="data/msmarco/2019qrels-pass_ms_marco.txt" 

QUERY_DOCS_PATH="data/msmarco/msmarco-passagetest2019-top1000.tsv.sorted_43"

#MODEL_PATH="experiments/l1_0_Emb_bert_Sparse_1000_bsz_512_lr_0.0001_TF_L_2_H_4_D_768_P_AVG"
EXPERIMENTS='experiments'

cd ..

for MODEL_PATH in ${EXPERIMENTS}/*/ ; do
	if [[ ! $MODEL_PATH = *Dev* ]]; then
		echo "${MODEL_PATH}"
		python3 top_k_analysis.py model_path=$MODEL_PATH batch_size=$BATCH_SIZE
	fi
done
