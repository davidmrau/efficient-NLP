
<<<<<<< HEAD
POINTWISE=True

BATCH_SIZE="128"

TRAIN_SAMPLES="100000"
VAL_SAMPLES="10000"
NUM_EPOCHS="10000"
PATIENCE=5

# load complete bert
LOAD_LAYERS="0-1-2-3-4-5-6-7-8-9-10-11-12"

# LOAD_MODEL_PATH="/project/draugpu/data/msmarco_rel_embed/best_model_cpu.model"
LOAD_MODEL_PATH="default"

LR="0.000001"

MSMARCO_QUERIES_TRAIN_PATH="/project/draugpu/data/msmarco/queries.train.tsv_bert_stop_none_remove_unk.tsv"
MSMARCO_COLLECTION_PATH="/project/draugpu/data/msmarco/collection.tsv_bert_stop_none_remove_unk.tsv"

python3 main.py dataset=msmarco model=bert batch_size_train=$BATCH_SIZE samples_per_epoch_train=$TRAIN_SAMPLES \
	samples_per_epoch_val=$VAL_SAMPLES bert.load_bert_layers=$LOAD_LAYERS max_samples_per_gpu=-1 \
	patience=$PATIENCE lr=$LR num_workers=1 num_epochs=$NUM_EPOCHS bert.load_bert_path=$LOAD_MODEL_PATH \
	bert.point_wise=$POINTWISE msmarco_query_train=$MSMARCO_QUERIES_TRAIN_PATH msmarco_docs_train=$MSMARCO_COLLECTION_PATH

=======
BATCH_SIZE="512"

MODEL=tf


# Experiment parameters:
EXPERIMENT_FOLDER='experiments/'


# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="4"
# TF_HID_DIMS="256 512"
TF_POOLS="AVG"
EMBEDDINGS="bert"

L1_SCALARS="0"


STOPWORDS='none'

TRAIN_SAMPLES="100000"
VAL_SAMPLES="10000"
NUM_EPOCHS="100000"


SPARSE_DIMENSIONS="5000"

python3 main.py dataset=msmarco model=bert batch_size_train=100 batch_size_test=100 samples_per_epoch_train=100 samples_per_epoch_val=10 bert.load_bert_layers=0-1-2-3-4 max_samples_per_gpu=-1


# for EMBEDDING in ${EMBEDDINGS}; do
# 	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
# 		for L1_SCALAR in ${L1_SCALARS}; do

# 			for TF_LAYER in ${TF_LAYERS}; do
# 				for TF_HEAD in ${TF_HEADS}; do
# 					for TF_POOL in ${TF_POOLS}; do

# 						python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
# 						tf.num_of_layers=${TF_LAYER} tf.num_attention_heads=${TF_HEAD} tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL} dataset=msmarco \
# 						samples_per_epoch_train=${TRAIN_SAMPLES} samples_per_epoch_val=${VAL_SAMPLES} stopwords=${STOPWORDS} num_epochs=${NUM_EPOCHS} num_workers=1 \
# 						num_workers=1 patience=10

# 					done
# 				done
# 			done
# 		done
# 	done
# done

# python3 main.py model=tf batch_size=128 embedding="bert" sparse_dimensions=5000 l1_scalar=0 tf.num_of_layers=2 tf.num_attention_heads=4 tf.hidden_size=768 tf.pooling_method=AVG dataset=msmarco samples_per_epoch_train=1000 samples_per_epoch_val=100 stopwords=none num_epochs=2 num_workers=1 num_workers=1 patience=100 layer_norm=False
>>>>>>> origin/master
