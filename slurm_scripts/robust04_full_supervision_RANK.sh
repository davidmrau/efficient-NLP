DATASET='robust04'
STOPWORDS='lucene'
MODEL='rank'
EMBEDDING="glove"
BATCH_SIZE=128
TRAIN_SAMPLES="1000"
VAL_SAMPLES="30"
NUM_EPOCHS="10000"


HIDDEN="32"


SAMPLER='uniform'

SAMPLES_PER_QUERY=-1




echo 'running'
python3 main_full_supervision.py model=${MODEL} batch_size_test=${BATCH_SIZE} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} \
rank_model.hidden_sizes=${HIDDEN} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 samples_per_query=${SAMPLES_PER_QUERY} sampler=${SAMPLER} \
num_workers=1 patience=20 rank_model.weights='uniform' lr=0.001

