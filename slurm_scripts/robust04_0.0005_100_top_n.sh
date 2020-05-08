#!/bin/bash
# Set job requirements
#SBATCH --job-name=0.5_100_top_n
#SBATCH --ntasks=1
#SBATCH --partition=gpu_shared
#SBATCH --time=120:00:00
#SBATCH --mem=100G


#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

BATCH_SIZE="16"

cd ..
DATASET='robust04'
STOPWORDS='lucene'

MODEL=snrm
EMBEDDINGS="glove"

TRAIN_SAMPLES="100000"
VAL_SAMPLES="10000"
NUM_EPOCHS="10000"


SPARSE_DIMENSIONS="5000"
N_GRAM_MODELS="cnn"
SNRM_HIDDEN="100-300"


L1_SCALARS="0.0005"

SAMPLES_PER_QUERY=100

SAMPLER="top_n"



bash telegram.sh "${0} ${SLURM_JOBID} Started"


for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
				snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
				samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 samples_per_query=${SAMPLES_PER_QUERY} sampler=${SAMPLER} \
				num_workers=1 patience=10

			done
		done
	done
done


bash telegram.sh "${0} ${SLURM_JOBID} Finished"
