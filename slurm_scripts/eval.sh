#!/bin/bash
# Set job requirements
#SBATCH --job-name=eval
#SBATCH --ntasks=1
#SBATCH --partition=gpu_shared
#SBATCH --time=01:00:00
#SBATCH --mem=100G

BATCH_SIZE="256"

#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176


cd ..

QRELS_PATH="data/msmarco/2019qrels-pass_ms_marco.txt" 

QUERY_DOCS_PATH="data/msmarco/msmarco-passagetest2019-top1000.tsv.sorted_43"

#MODEL_PATH="experiments/l1_0_Emb_bert_Sparse_1000_bsz_512_lr_0.0001_TF_L_2_H_4_D_768_P_AVG"
MODEL_PATH='experiments/l1_1_Emb_bert_Sparse_5000_bsz_128_lr_0.0001_TF_L_2_H_4_D_768_P_AVG_ACT_delu/'


python3 inference.py model_path=$MODEL_PATH qrels=$QRELS_PATH q_docs=$QUERY_DOCS_PATH batch_size=$BATCH_SIZE MaxMRRRank=10



MODEL_PATH='experiments/l1_0_Emb_bert_Sparse_5000_bsz_128_lr_0.0001_TF_L_2_H_4_D_768_P_AVG_ACT_delu/'


python3 inference.py model_path=$MODEL_PATH qrels=$QRELS_PATH q_docs=$QUERY_DOCS_PATH batch_size=$BATCH_SIZE MaxMRRRank=10

