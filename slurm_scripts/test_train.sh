
QUERY_FILE='msmarco-test2019-queries-43-judged.tokenized.tsv'
QRELS='2019qrels-pass_filtered_ms_marco.txt'
DOCS_FILE='msmarco-passagetest2019-top1000_43.tsv.d_id_doc.tokenized_uniq.tsv'


cd ..
# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'

EMBEDDINGS="bert"
SPARSE_DIMENSIONS="400"
L1_SCALARS="1"
BATCH_SIZES="64"

# Model parameters:
MODELS="snrm"
# SNRM parameters:
SNRM_HIDDENS="200"

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
					python3 main.py model_folder=${EXP_DIR} query_file=${QUERY_FILE} qrels=${QRELS} docs_file=${DOCS_FILE} debug=True

				done
			done
		done
	done
done
