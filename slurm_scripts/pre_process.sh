#!/bin/sh
#SBATCH --job-name=download_docs_all
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=24:00:00
cd ..

# tokenize
python3 tokenizer.py --folder data/msmarco/ --fname collection.tsv --max_len 150 
python3 tokenizer.py --folder data/msmarco/ --fname queries.dev.tsv 
python3 tokenizer.py --folder data/msmarco/ --fname queries.eval.tsv
python3 tokenizer.py --folder data/msmarco/ --fname queries.train.tsv 

# create debug triplets file

head -100000 data/msmarco/qidpidtriples.train.full.tsv >> qidpidtriples.train.full.debug.tsv

# calculate offset dicts

python3 offset_dict.py --fname data/msmarco/qidpidtriples.train.full.tsv --delimiter '\t' --line_index_is_id
python3 offset_dict.py --fname data/msmarco/qidpidtriples.train.full.debug.tsv --delimiter '\t' --line_index_is_id
python3 offset_dict.py --fname data/msmarco/collection.tokenized.tsv
python3 offset_dict.py --fname data/msmarco/queries.dev.tokenized.tsv
python3 offset_dict.py --fname data/msmarco/queries.eval.tokenized.tsv
python3 offset_dict.py --fname data/msmarco/queries.train.tokenized.tsv
