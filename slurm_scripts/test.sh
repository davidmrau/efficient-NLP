#!/bin/bash
# Set job requirements
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --partition=normal

#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=kondilidisn9@gmail.com

#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

BATCH_SIZE="64"

cd ..
DATASET='msmarco'
STOPWORDS='none'

MODEL=tf

EMBEDDINGS="bert"
L1_SCALARS="0"
SPARSE_DIMENSIONS="5000"
N_GRAM_MODELS="cnn"

# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="8"
# TF_HID_DIMS="256 512"
TF_POOLS="AVG"

# Dense EMB-500-100-500
# SNRM_HIDDEN="100-300"

for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				
				python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} \
				l1_scalar=${L1_SCALAR} samples_per_epoch_train=1000 samples_per_epoch_val=128 stopwords=${STOPWORDS} num_epochs=1 num_workers=0 \
				dataset=${DATASET} tf.load_bert_layers=0-1-2-3-4-5-6-7-8-9-10-11-12 tf.load_bert_path="experiments_msmarco/lm_batch_8/best_model_cpu.model" \
				tf.num_of_layers=2 tf.num_attention_heads=2 tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL} debug=True tf.pooling_method=AVG
				
			done
		done
	done
done


