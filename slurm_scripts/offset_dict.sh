#!/bin/sh
#SBATCH --job-name=train_snrm
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=01:00:00
cd ..
#python3 offset_dict.py --fname data/msmarco/qidpidtriples.train.full.tsv --delimiter '\t' line_index_is_id
#python3 offset_dict.py --fname data/msmarco/collection.tokenized.tsv
python3 offset_dict.py --fname data/msmarco/queries.dev.tokenized.tsv
python3 offset_dict.py --fname data/msmarco/queries.eval.tokenized.tsv
python3 offset_dict.py --fname data/msmarco/queries.train.tokenized.tsv
