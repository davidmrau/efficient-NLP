DATASET='robust04'
STOPWORDS='lucene'
MODEL='rank'
EMBEDDING="glove"
TRAIN_SAMPLES="10000000"
VAL_SAMPLES="10000"
NUM_EPOCHS="10000"


HIDDENS="512-512"
LRS="0.00005"
PATIENCE=25


BATCH_SIZE=1024

echo 'running'
for HIDDEN in $HIDDENS;do
	for LR in $LRS; do
		echo $LR $HIDDEN
		python3 main.py model=${MODEL} batch_size_test=${BATCH_SIZE} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} rank_model.hidden_sizes=${HIDDEN} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 patience=$PATIENCE rank_model.weights='uniform' lr=$LR  rank_model.trainable_weights=True rank_model.model_type='interaction-based' telegram=True
	done
done
