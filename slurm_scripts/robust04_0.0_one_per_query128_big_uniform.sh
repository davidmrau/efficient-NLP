
BATCH_SIZE="128"

DATASET='robust04'
STOPWORDS='lucene'

MODEL=snrm
EMBEDDINGS="glove"

TRAIN_SAMPLES="500000"
VAL_SAMPLES="20000"
NUM_EPOCHS="100000"


SPARSE_DIMENSIONS="5000"
N_GRAM_MODELS="cnn"
SNRM_HIDDEN="300-500"


L1_SCALARS="0"

SAMPLES_PER_QUERY=-1

SAMPLER="uniform"



for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

				python3 main.py model=${MODEL} batch_size_train=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
				snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
				samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 samples_per_query=${SAMPLES_PER_QUERY} sampler=${SAMPLER} \
				num_workers=1 patience=100 single_sample=True telegram=True

			done
		done
	done
done


