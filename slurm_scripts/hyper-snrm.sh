
BATCH_SIZE="256"

cd ..


MODEL=snrm


# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'

      # All dense experiments
echo 'Dense Experiments :'
  # Bert Embeddings (768)
EMBEDDING="bert"
L1_SCALAR="0"
# Dense EMB-300-100-300
SPARSE_DIMENSION="300"
SNRM_HIDDEN="300-100"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}


# Dense EMB-500-100-500
SPARSE_DIMENSION="500"
SNRM_HIDDEN="500-100"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}


# Dense 300-100
SPARSE_DIMENSION="100"
SNRM_HIDDEN="300"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}


# Dense 500-300-100-300-500
SPARSE_DIMENSION="500"
SNRM_HIDDEN="500-300-100-300"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}



  # Bert Embeddings (300)
EMBEDDING="glove"

SPARSE_DIMENSION="300"
SNRM_HIDDEN="100"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}


SPARSE_DIMENSION="500"
SNRM_HIDDEN="100-300"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}


SPARSE_DIMENSION="500"
SNRM_HIDDEN="100"

MODEL_STRING=${MODEL}_${SNRM_HIDDEN}
EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}
echo ${EXP_DIR}
# echo "Training"
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
model_folder=${EXP_DIR}



echo "Sparse experiments :"

EMBEDDING="bert"
# Sparse EMB-300-100-300- {1000/5000/10000} , l1 {1,2}
SPARSE_DIMENSIONS="1000 5000 10000"
L1_SCALARS="1 2"
SNRM_HIDDENS="300-100-300 500-100-500 500-300-100-300-500"


for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
	for L1_SCALAR in ${L1_SCALARS}; do

			MODEL=snrm

			for SNRM_HIDDEN in ${SNRM_HIDDENS}; do
				MODEL_STRING=${MODEL}_${SNRM_HIDDEN}

				EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}

				echo ${EXP_DIR}
				# echo "Training"
				python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
				model_folder=${EXP_DIR}

			done
	done
done




EMBEDDING="glove"
# Sparse EMB-300-100-300- {1000/5000/10000} , l1 {1,2}
SPARSE_DIMENSIONS="1000 5000 10000"
L1_SCALARS="1 2"
SNRM_HIDDENS="100-300 100-300-500 100-500"


for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
	for L1_SCALAR in ${L1_SCALARS}; do

			MODEL=snrm

			for SNRM_HIDDEN in ${SNRM_HIDDENS}; do
				MODEL_STRING=${MODEL}_${SNRM_HIDDEN}

				EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}

				echo ${EXP_DIR}
				# echo "Training"
				python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
				model_folder=${EXP_DIR}

			done
	done
done
