
# triplet sampling options:

SAMPLES_PER_QUERY=-1

SAMPLER="zipf"



# training parameters :

BATCH_SIZE="128"

TRAIN_SAMPLES="500000"
VAL_SAMPLES="20000"
NUM_EPOCHS="100000"


# 0.2 -> 20% of neurons' acivations are set to 0
DROPOUT="0.2"

LEARNING_RATE="0.0001"


# Architecture and weight parameters of the rank model :

# input dims : 2 * 300 .

# one hidden layer of 300 neurons
# HIDDEN="300"

# a first hidden layer of 300 neurons, followed by a second hidden layer of 100 neurons
HIDDEN="300-100"


# weights of rank model :
WEIGHTS='uniform' # a path to a pickled tensor OR "uniform" OR "random" OR 'data/embeddings/idf_tensor.p' -> [IDF values] 
TRAINABLE_WEIGHTS="True"

python3 main.py model=rank batch_size_train=${BATCH_SIZE} embedding="glove" \
rank_model.hidden_sizes=${HIDDEN} rank_model.weights=${WEIGHTS} rank_model.trainable_weights=${TRAINABLE_WEIGHTS} \
dataset='robust04' samples_per_epoch_train=${TRAIN_SAMPLES} weak_min_results=10 \
samples_per_epoch_val=${VAL_SAMPLES} stopwords='lucene' num_epochs=${NUM_EPOCHS} num_workers=1 samples_per_query=${SAMPLES_PER_QUERY} sampler=${SAMPLER} \
num_workers=1 patience=100 single_sample=True max_samples_per_gpu=-1 lr=${LEARNING_RATE} rank_model.dropout_p=${DROPOUT}
