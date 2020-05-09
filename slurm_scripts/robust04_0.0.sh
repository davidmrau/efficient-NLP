
BATCH_SIZE="64"

DATASET='robust04'
STOPWORDS='lucene'

MODEL=snrm
EMBEDDINGS="glove"

TRAIN_SAMPLES="10000"
VAL_SAMPLES="10000"
NUM_EPOCHS="10000"

L1_SCALARS="0"
SPARSE_DIMENSIONS="5000"


N_GRAM_MODELS="cnn"

LRS="0.001 0.00001"


# Dense EMB-500-100-500
SNRM_HIDDEN="100-300"
bash telegram.sh "${0} ${SLURM_JOBID} Started"
for EMBEDDING in ${EMBEDDINGS}; do

	for L1_SCALAR in ${L1_SCALARS}; do
		for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
			for N_GRAM_MODEL in ${N_GRAM_MODELS}; do
				for LR in ${LRS}; do
					python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} snrm.hidden_sizes=${SNRM_HIDDEN} n_gram_model=${N_GRAM_MODEL} dataset=${DATASET} samples_per_epoch_train=${TRAIN_SAMPLES} samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=2 lr=${LR}
				done

			done
		done
	done
done

bash telegram.sh "${0} ${SLURM_JOBID} Done"
