#!/bin/sh
#SBATCH --job-name=eval_snrm
#SBATCH --nodes=1
#SBATCH -p gpu_shared 
#SBATCH --time=4:00:00


cd ..
QUERY_FILE='msmarco-test2019-queries-43-judged.tokenized.tsv'
QRELS='2019qrels-pass_without_q_tabs.txt'


MODEL_DIR='experiments/model_snrm_l1_scalar_1_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False'
python3 online_inference.py model_folder=${MODEL_DIR} query_file=${QUERY_FILE} qrels=${QRELS}
