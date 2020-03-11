#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1
#SBATCH -p gpu_shared 
#SBATCH --time=4:00:00
cd ..
python3 create_index.py model_folder=experiments/model_snrm_l1_scalar_1_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False/ docs_file=msmarco-passagetest2019-top1000_43.tsv.d_id_doc.tokenized_uniq.tsv
