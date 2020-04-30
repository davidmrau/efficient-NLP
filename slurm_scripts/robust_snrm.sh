#!/bin/bash
# Set job requirements
#SBATCH --job-name=robust_snrm
#SBATCH --ntasks=1
#SBATCH --partition=gpu_short
##SBATCH --gres=gpu:4
##SBATCH --time=120:00:00
#SBATCH --mem=100G
cd ..
#python3 main.py model=snrm embedding=glove dataset=robust04 samples_per_epoch=100000 log_every_ratio=0.01 stopwords=lucene batch_size=64 num_workers=1
#python3 main.py model=snrm embedding=glove dataset=robust04 samples_per_epoch=100000 log_every_ratio=0.01 stopwords=lucene batch_size=32 num_workers=1
python3 main.py model=snrm embedding=glove dataset=robust04 samples_per_epoch=100000 log_every_ratio=0.01 stopwords=lucene batch_size=4 num_workers=1
python3 main.py model=snrm embedding=bert dataset=robust04 samples_per_epoch=100000 log_every_ratio=0.01 stopwords=lucene batch_size=64 num_workers=1
python3 main.py model=snrm embedding=bert dataset=robust04 samples_per_epoch=100000 log_every_ratio=0.01 stopwords=lucene batch_size=32 num_workers=1
python3 main.py model=snrm embedding=bert dataset=robust04 samples_per_epoch=100000 log_every_ratio=0.01 stopwords=lucene batch_size=16 num_workers=1
