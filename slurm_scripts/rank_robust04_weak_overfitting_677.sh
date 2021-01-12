
BATCH_SIZE="128"

DATASET='robust04'
STOPWORDS='lucene'

MODEL=rank
EMBEDDING="glove"

TRAIN_SAMPLES="10000"
VAL_SAMPLES="100"
NUM_EPOCHS="100"

# rank model's parameters
HIDDEN="256-256"
WEIGHTS="uniform"
TRAINABLE_WEIGHTS="True"


L1_SCALARS="0"

QUERIES_TEST='data/robust04/weak_overfitting/677/test_query.csv'
QUERIES_TRAIN='data/robust04/weak_overfitting/677/train_query.csv'
TOP_1000='data/robust04/weak_overfitting/677/top_1000.csv_TRIPLETS'
TEST_TOP_2000='data/robust04/weak_overfitting/677/test_top_2000.csv'



SAMPLER="uniform"

MARGIN=1


python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} rank_model.weights=$WEIGHTS \
rank_model.hidden_sizes=${HIDDEN} rank_model.trainable_weights=$TRAINABLE_WEIGHTS dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} sampler=${SAMPLER} \
num_workers=1 patience=1000  robust_ranking_results_test=$TEST_TOP_2000 robust_query_test=$QUERIES_TEST \
robust_triples=$TOP_1000 robust_query_train=$QUERIES_TRAIN weak_overfitting_test=True margin=$MARGIN add="677" max_samples_per_gpu=380  sample_random=False single_sample=True
