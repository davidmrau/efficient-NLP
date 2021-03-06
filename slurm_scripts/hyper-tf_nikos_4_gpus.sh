#!/bin/bash
# Set job requirements
#SBATCH --job-name=test
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=120:00:00
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


MODEL=tf

BATCH_SIZE="512"

cd ..

# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'


# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="4"
# TF_HID_DIMS="256 512"
TF_POOLS="CLS AVG"
EMBEDDINGS="bert"


      # All sparse experiments
echo 'Sparse TF Experiments :'
L1_SCALARS="1 0.1 2"

SPARSE_DIMENSIONS="500 300 1000 5000 10000"




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


							MODEL_STRING=${MODEL}_L_${TF_LAYER}_H_${TF_HEAD}_D_${TF_HID_DIM}_P_${TF_POOL}

							EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}

							echo ${EXP_DIR}
							# echo "Training"
							python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
							tf.num_of_layers=${TF_LAYER} tf.num_attention_heads=${TF_HEAD} tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL}  \
							model_folder=${EXP_DIR}

						done
					done
				done
			done
		done
	done
done
