

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
LRS="0.001 0.00005"

L1_SCALARS="0"

SAMPLES_PER_QUERY=-1

SAMPLER="uniform"



for EMBEDDING in ${EMBEDDINGS}; do
	for LR in ${LRS};do
		for L1_SCALAR in ${L1_SCALARS}; do
			for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
				for N_GRAM_MODEL in ${N_GRAM_MODELS}; do

					python3 main.py model=${MODEL} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
					snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} \
					samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 samples_per_query=${SAMPLES_PER_QUERY} sampler=${SAMPLER} \
					num_workers=1 patience=1000 lr=${LR}
	
				done
			done
		done
	done
done


