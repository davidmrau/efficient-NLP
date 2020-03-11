#!/bin/sh
#SBATCH --job-name=eval_snrm
#SBATCH --nodes=1
#SBATCH -p gpu_shared 
#SBATCH --time=4:00:00


cd ..
QUERY_FILE='msmarco-test2019-queries_43.tokenized.tsv'
QRELS='2019qrels-pass_filtered_ms_marco.txt'
#DOCS_FILE='msmarco-passagetest2019-top1000.tsv.d_id_doc.tokenized_uniq.tsv'
DOCS_FILE='msmarco-passagetest2019-top1000_43.tokenized.tsv'
#MODEL_DIR='experiments/model_snrm_l1_scalar_0.0_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False'
#MODEL_DIR='experiments/model_snrm_l1_scalar_1_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False'
MODEL_DIR='experiments/model_tf_l1_scalar_0.0_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False'
python3 inference.py model_folder=${MODEL_DIR} query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE}



# run trec
./trec_eval -l 2 data/msmarco/${QRELS} ${MODEL_DIR}/ranking_results.${QUERY_FILE}.trec
