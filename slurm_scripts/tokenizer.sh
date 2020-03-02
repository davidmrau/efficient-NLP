#!/bin/sh
#SBATCH --job-name=download_docs_all
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=24:00:00
cd ..
#python tokenizer.py --folder data/msmarco/ --fname collection.tsv --max_len 150 
#python tokenizer.py --folder data/msmarco/ --fname queries.dev.tsv 
#python tokenizer.py --folder data/msmarco/ --fname 'queries.eval.tsv' 
python tokenizer.py --folder data/msmarco/ --fname queries.train.tsv 
