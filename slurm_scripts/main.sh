



# DOCS_FILE='top1000.tsv.d_id_doc.tokenized.tsv'
# QUERY_FILE='msmarco-test2019-queries.tokenized.tsv'
# QRELS='qrels.train.tsv'

QUERY_FILE='msmarco-test2019-queries_43.tokenized.tsv'
QRELS='2019qrels-pass_filtered_ms_marco.txt'
DOCS_FILE='msmarco-passagetest2019-top1000_43.tokenized.tsv'


cd ..
# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'

EMBEDDINGS="bert"
SPARSE_DIMENSIONS="10000"
L1_SCALARS="0 1 2"
BATCH_SIZES="256"

# Model parameters:
MODELS="snrm tf"
# SNRM parameters:
SNRM_HIDDENS="100-300 300-500 300-100-3000"
# Transformer parameters:
TF_LAYERS="2 4"
TF_HEADS="4 8"
TF_HID_DIMS="256 512"
TF_POOLS="CLS"


for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do
			for BATCH_SIZE in ${BATCH_SIZES}; do

				MODEL=snrm

				for SNRM_HIDDEN in ${SNRM_HIDDENS}; do
					MODEL_STRING=${MODEL}_${SNRM_HIDDEN}

					EXP_DIR=${EXPERIMENT_FOLDER}l1_${L1_SCALAR}_Emb_${EMBEDDING}_Sparse_${SPARSE_DIMENSION}_bsz_${BATCH_SIZE}_${MODEL_STRING}

					echo ${EXP_DIR}
					# echo "Training"
					python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} \
					model_folder=${EXP_DIR} query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE}

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

				if [ $EMBEDDING==bert ]; then
				  TF_HID_DIMS=768
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
								model_folder=${EXP_DIR} query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE}

							done
						done
					done
				done
			done
		done
	done
done
