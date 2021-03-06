
BATCH_SIZE="128"

DATASET='robust04'
STOPWORDS='lucene'

MODEL=snrm
EMBEDDINGS="glove"

TRAIN_SAMPLES="1000"
VAL_SAMPLES="1000"
NUM_EPOCHS="100"


SPARSE_DIMENSIONS="1000"
N_GRAM_MODELS="cnn"
SNRM_HIDDEN="100-300"


L1_SCALARS="0"

QUERIES_TEST='data/robust04/weak_overfitting/301/test_query.csv'
QUERIES_TRAIN='data/robust04/weak_overfitting/301/train_query.csv'
TOP_2000='data/robust04/weak_overfitting/301/test_top_2000.csv'
TEST_TOP_2000='data/robust04/weak_overfitting/301/top_2000.csv'


SAMPLER="uniform"



for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
				snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
				samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} sampler=${SAMPLER} \
				num_workers=0 patience=1000 single_sample=True  robust_ranking_results_test=$TEST_TOP_2000 robust_query_test=$QUERIES_TEST \
				robust_ranking_results_train=$TOP_2000 robust_query_train=$QUERIES_TRAIN weak_overfitting_test=True add="301"

			done
		done
	done
done


SAMPLER="zipf"



for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
				snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
				samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} sampler=${SAMPLER} \
				num_workers=0 patience=1000 single_sample=True  robust_ranking_results_test=$TEST_TOP_2000 robust_query_test=$QUERIES_TEST \
				robust_ranking_results_train=$TOP_2000 robust_query_train=$QUERIES_TRAIN weak_overfitting_test=True add="301"

			done
		done
	done
done


SAMPLER="linear"



for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
				snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
				samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} sampler=${SAMPLER} \
				num_workers=0 patience=1000 single_sample=True  robust_ranking_results_test=$TEST_TOP_2000 robust_query_test=$QUERIES_TEST \
				robust_ranking_results_train=$TOP_2000 robust_query_train=$QUERIES_TRAIN weak_overfitting_test=True add="301"

			done
		done
	done
done
