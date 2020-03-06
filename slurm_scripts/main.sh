#
# cd ..
#
# DOCS_FILE=top1000.tsv.d_id_doc.tokenized.tsv
# QUERY_FILE=msmarco-test2019-queries.tokenized.tsv
# QRELS=qrels.train.tsv
#
#
# EXPERIMENT_FOLDER='experiments/'
#
# echo ${QRELS}
# echo ${QUERY_FILE}
# echo ${DOCS_FILE}
# for HIDDEN in 100 200;do
# 	MODEL_FOLDER=${EXPERIMENT_FOLDER}${HIDDEN}_model
# 	MODEL_FOLDER='experiments/model_snrm_l1_scalar_1_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False/'
# 	echo ${MODEL_FOLDER}
# 	python3 main.py model_folder=${MODEL_FOLDER}
# 	python3 create_index.py model_folder=${MODEL_FOLDER} docs_file=${DOCS_FILE}
# 	python3 online_inference.py model_folder=${MODEL_FOLDER} query_file=${QUERY_FILE} qrels=${QRELS}
# done

# Experiment parameters:
NUM_WORKERS=7

EMBEDDINGS="bert"
SPARSE_DIMENSIONS="10000"
L1_SCALARS="0 1 2"
BATCH_SIZES="64 256"

# Model parameters:
MODELS="snrm tf"

SNRM_HIDDENS="100-300 100-500"

TF_LAYERS="1 2 4"
TF_HEADS="2 4 8"
TF_HID_DIMS="128 256 512"
TF_POOLS="CLS MAX AVG"


for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do
			for BATCH_SIZE in ${BATCH_SIZES}; do

				MODEL=snrm

				for SNRM_HIDDEN in ${SNRM_HIDDENS}; do
					MODEL_STRING=${MODEL}_${SNRM_HIDDEN}

					EXP_DIR=l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}

					echo ${EXP_DIR}

				done
			done
		done
	done
done


for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do
			for BATCH_SIZE in ${BATCH_SIZES}; do

				MODEL=tf

				for TF_LAYER in ${TF_LAYERS}; do
					for TF_HEAD in ${TF_HEADS}; do
						for TF_HID_DIM in ${TF_HID_DIMS}; do
							for TF_POOL in ${TF_POOLS}; do


								MODEL_STRING=${MODEL}_L_${TF_LAYER}_H_${TF_HEAD}_D_${TF_HID_DIM}_P_${TF_POOL}

								EXP_DIR=l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}

								echo ${EXP_DIR}


							done
						done
					done
				done
			done
		done
	done
done

#
# if ${MODEL} = snrm
# then
#   MODEL_STRING=${MODEL}_layers_{}
# else
#   STATEMENTS2
# fi
