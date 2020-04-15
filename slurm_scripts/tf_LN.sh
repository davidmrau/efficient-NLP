#!/bin/bash
# Set job requirements
#SBATCH --job-name=tf_LN
#SBATCH --ntasks=1
#SBATCH --partition=gpu_shared
#SBATCH --time=60:00:00
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

BATCH_SIZES="128"



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

L1_SCALARS="0 1"


SPARSE_DIMENSIONS="5000"
LARGE_OUT_BIASES_OPTIONS="False True"

LAST_LAYER_NORM_OPTIONS="True"

last_layer_norm

for LAST_LAYER_NORM in ${LAST_LAYER_NORM_OPTIONS}; do 

	for BATCH_SIZE in ${BATCH_SIZES}; do 
		
		for EMBEDDING in ${EMBEDDINGS}; do
			
			for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
				
				for L1_SCALAR in ${L1_SCALARS}; do
					

					for TF_LAYER in ${TF_LAYERS}; do
						
						for TF_HEAD in ${TF_HEADS}; do
							
							
							for TF_POOL in ${TF_POOLS}; do
								

								for LARGE_OUT_BIASES in ${LARGE_OUT_BIASES_OPTIONS}; do

									python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
									tf.num_of_layers=${TF_LAYER} tf.num_attention_heads=${TF_HEAD} tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL} large_out_biases=${LARGE_OUT_BIASES} tf.last_layer_norm=${LAST_LAYER_NORM}

								done
							done
						done
					done
				done
			done
		done
	done
done