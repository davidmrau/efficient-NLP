
BATCH_SIZE="128"

DATASET='robust04'
STOPWORDS='lucene'

MODEL=rank
EMBEDDING="glove"

TRAIN_SAMPLES="1000"
VAL_SAMPLES="1000"
NUM_EPOCHS="100"

# rank model's parameters
HIDDEN="128-128"
WEIGHTS="uniform"
TRAINABLE_WEIGHTS="True"


L1_SCALARS="0"

QUERIES_TEST='data/robust04/weak_overfitting/695/test_query.csv'
QUERIES_TRAIN='data/robust04/weak_overfitting/695/train_query.csv'
TOP_2000='data/robust04/weak_overfitting/695/test_top_2000.csv'
TEST_TOP_2000='data/robust04/weak_overfitting/695/top_2000.csv'


SAMPLER="uniform"


python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} rank_model.weights=$WEIGHTS \
rank_model.hidden_sizes=${HIDDEN} rank_model.trainable_weights=$TRAINABLE_WEIGHTS dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} sampler=${SAMPLER} \
num_workers=0 patience=1000 single_sample=True  robust_ranking_results_test=$TEST_TOP_2000 robust_query_test=$QUERIES_TEST \
robust_ranking_results_train=$TOP_2000 robust_query_train=$QUERIES_TRAIN weak_overfitting_test=True add="695"
