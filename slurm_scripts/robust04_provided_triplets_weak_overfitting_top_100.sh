
BATCH_SIZE="128"

DATASET='robust04'
STOPWORDS='lucene'

MODEL=snrm
EMBEDDING="glove"

TRAIN_SAMPLES="100000"
VAL_SAMPLES="10000"
NUM_EPOCHS="10000"


SPARSE_DIMENSION="1000"
N_GRAM_MODEL="cnn"
SNRM_HIDDEN="100-300"


L1_SCALAR="0"

QUERIES_TEST='data/robust04/weak_overfitting/all/test_query.csv'
QUERIES_TRAIN='data/robust04/weak_overfitting/all/train_query.csv'
TOP_2000='data/robust04/weak_overfitting/all/test_top_2000.csv'
TEST_TOP_2000='data/robust04/weak_overfitting/all/top_2000.csv'


python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} sampler=${SAMPLER} \
num_workers=1 patience=100000 single_sample=True  robust_ranking_results_test=$TEST_TOP_2000 robust_query_test=$QUERIES_TEST \
robust_ranking_results_train=$TOP_2000 robust_query_train=$QUERIES_TRAIN weak_overfitting_test=True add="top_100" provided_triplets=True

