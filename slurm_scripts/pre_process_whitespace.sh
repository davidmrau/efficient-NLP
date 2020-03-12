#!/bin/sh
#SBATCH --job-name=pre_processing
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=03:00:00
cd ..

# tokenize
python3 tokenizer.py --folder data/msmarco/ --fname collection.tsv --max_len 150 --whitespace
python3 tokenizer.py --folder data/msmarco/ --fname queries.dev.tsv --whitespace 
python3 tokenizer.py --folder data/msmarco/ --fname queries.eval.tsv --whitespace
python3 tokenizer.py --folder data/msmarco/ --fname queries.train.tsv --whitespace


# calculate offset dicts

python3 offset_dict.py --fname data/msmarco/collection.tokenized.white.tsv
python3 offset_dict.py --fname data/msmarco/queries.dev.tokenized.white.tsv
python3 offset_dict.py --fname data/msmarco/queries.eval.tokenized.white.tsv
python3 offset_dict.py --fname data/msmarco/queries.train.tokenized.white.tsv


