#!/bin/sh
#SBATCH --job-name=pre_pro
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --time=12:00:00


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

# tokenize

python3 tokenizer.py --input_file=data/robust04/AOL-queries-all_filtered.txt.names --tokenizer=glove --remove_unk --stopwords=lucene --max_len 1500
python3 tokenizer.py --input_file=data/robust04/robust04_raw_docs.num_query --tokenizer=glove --remove_unk --stopwords=lucene --max_len 1500
python3 tokenizer.py --input_file=data/robust04/04.testset_num_query_lower --tokenizer=glove --remove_unk --stopwords=lucene --max_len 1500


python3 tokenizer.py --input_file=data/robust04/AOL-queries-all_filtered.txt.names --tokenizer=bert --remove_unk --stopwords=lucene --max_len 1500
python3 tokenizer.py --input_file=data/robust04/robust04_raw_docs.num_query --tokenizer=bert --remove_unk --stopwords=lucene --max_len 1500
python3 tokenizer.py --input_file=data/robust04/04.testset_num_query_lower --tokenizer=bert --remove_unk --stopwords=lucene --max_len 1500




python3 offset_dict.py --fname data/robust04/robust04_raw_docs.num_query_glove_stop_lucene_remove_unk_max_len_1500.tsv
python3 offset_dict.py --fname data/robust04/AOL-queries-all_filtered.txt.names_glove_stop_lucene_remove_unk_max_len_1500.tsv
python3 offset_dict.py --fname data/robust04/04.testset_num_query_lower_glove_stop_lucene_remove_unk_max_len_1500.tsv

python3 offset_dict.py --fname data/robust04/robust04_raw_docs.num_query_bert_stop_lucene_remove_unk_max_len_1500.tsv
python3 offset_dict.py --fname data/robust04/AOL-queries-all_filtered.txt.names_bert_stop_lucene_remove_unk_max_len_1500.tsv
python3 offset_dict.py --fname data/robust04/04.testset_num_query_lower_bert_stop_lucene_remove_unk_max_len_1500.tsv

