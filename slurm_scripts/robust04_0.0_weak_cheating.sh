BATCH_SIZE="16"

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

SAMPLER='uniform'
L1_SCALARS="0"

SAMPLES_PER_QUERY=-1





for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
				snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
				samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 samples_per_query=${SAMPLES_PER_QUERY} sampler=${SAMPLER} \
				num_workers=1 patience=10 margin=0 robust_ranking_results_train=data/${DATASET}/robust04_TREC_test_anserini_top_2000_qld_ranking_results add='weak_cheating' \
				robust_query_train=data/${DATASET}/04.testset_num_query_lower_${EMBEDDING}_stop_${STOPWORDS}_remove_unk_max_len_1500.tsv

			done
		done
	done
done



