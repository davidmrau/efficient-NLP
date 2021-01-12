
BATCH_SIZE="128"

TRAIN_SAMPLES="100000"
VAL_SAMPLES="10000"
NUM_EPOCHS="10000"
PATIENCE=5

# load complete bert
LOAD_LAYERS="0-1-2-3-4-5-6-7-8-9-10-11-12"
LOAD_LAYERS="0-1-2-3-4-5-6"

LOAD_MODEL_PATH="default"

LR="0.000001"


python3 main.py dataset=msmarco model=bert batch_size_train=$BATCH_SIZE samples_per_epoch_train=$TRAIN_SAMPLES \
	samples_per_epoch_val=$VAL_SAMPLES bert.load_bert_layers=$LOAD_LAYERS \
	patience=$PATIENCE lr=$LR num_workers=1 num_epochs=$NUM_EPOCHS bert.load_bert_path=$LOAD_MODEL_PATH add='random_input_0.1' rand_p=0.1 sub_batch_size=64

