#!/bin/bash
# Set job requirements
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --partition=gpu_short
#SBATCH --time=00:05:00
#SBATCH --mem=100G


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
DATASET='robust04'
STOPWORDS='lucene'

MODEL=snrm

EMBEDDINGS="glove"
L1_SCALARS="0"
SPARSE_DIMENSIONS="5000"
N_GRAM_MODELS="cnn"
LARGE_OUT_BIASES_OPTIONS="False"

# Dense EMB-500-100-500
SNRM_HIDDEN="100-300"

for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do
				for LARGE_OUT_BIASES in ${LARGE_OUT_BIASES_OPTIONS}; do
					python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=5000 samples_per_epoch_val=128 stopwords=${STOPWORDS} num_epochs=1 num_workers=2
				done
			done
		done
	done
done
