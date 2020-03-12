

QUERY_FILE='msmarco-test2019-queries_43.tokenized.tsv'
QRELS='2019qrels-pass_filtered_ms_marco.txt'
DOCS_FILE='msmarco-passagetest2019-top1000_43.tokenized.tsv'

MODEL=tf

BATCH_SIZES="256"

cd ..

# test what is the max batch size that can fit in the gpu. If the folloing trains at least one epoch, we are good for the rest experiments that follow
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=8 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=CLS  \
model_folder=Test_Batch_size_exp query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE}

# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'


# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="4"
# TF_HID_DIMS="256 512"
TF_POOLS="CLS AVG MAX"
EMBEDDINGS="bert glove"


      # All dense experiments
echo 'Dense Experiments :'
L1_SCALAR="0"
# Dense
SPARSE_DIMENSIONS="500 300"




for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do

		if [ $EMBEDDING==bert ]; then
		  TF_HID_DIMS=768
		fi

		if [ $EMBEDDING==glove ]; then
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
						model_folder=${EXP_DIR} query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE}

					done
				done
			done
		done
	done
done




      # All sparse experiments
echo 'Sparse TF Experiments :'
L1_SCALARS="1 2"

SPARSE_DIMENSIONS="500 300 1000 5000 10000"




for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do

			if [ $EMBEDDING==bert ]; then
			  TF_HID_DIMS=768
			fi

			if [ $EMBEDDING==glove ]; then
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
							model_folder=${EXP_DIR} query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE}

						done
					done
				done
			done
		done
	done
done
