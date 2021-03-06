#!/bin/bash
# Set job requirements
#SBATCH --job-name=bias_3
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
#SBATCH --mem=100G


#Loading modules
module purge
module load pre2019
module load eb
module load Python/3.6.3-foss-2017b
module load cuDNN/7.0.5-CUDA-9.0.176
module load NCCL/2.0.5-CUDA-9.0.176

BATCH_SIZE="256"



MODEL=tf


cd ..

# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'


# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="4"
# TF_HID_DIMS="256 512"
TF_POOLS="AVG"
EMBEDDINGS="bert"

L1_SCALARS="0"




SPARSE_DIMENSIONS="1000 5000 10000"

for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do

			if [ $EMBEDDING == bert ]; then
			  TF_HID_DIMS=768
			fi

			if [ $EMBEDDING == glove ]; then
			  TF_HID_DIMS=300
			fi


			for TF_LAYER in ${TF_LAYERS}; do
				for TF_HEAD in ${TF_HEADS}; do
					for TF_HID_DIM in ${TF_HID_DIMS}; do
						for TF_POOL in ${TF_POOLS}; do

							echo ${EXP_DIR}
							# echo "Training"
							python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
							tf.num_of_layers=${TF_LAYER} tf.num_attention_heads=${TF_HEAD} tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL} large_out_biases=True


						done
					done
				done
			done
		done
	done
done
