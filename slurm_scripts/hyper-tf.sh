
cd ..

BATCH_SIZE="1024"
# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="4 8"

TF_POOLS="AVG"
EMBEDDINGS="bert random"

L1_SCALARS="0 0.1 1"
# Dense
SPARSE_DIMENSIONS="1000 5000 10000"

TF_HID_DIM=768



for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do

			for TF_LAYER in ${TF_LAYERS}; do
				for TF_HEAD in ${TF_HEADS}; do
					for TF_POOL in ${TF_POOLS}; do

						# echo "Training"
						python3 main.py model=tf batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
						tf.num_of_layers=${TF_LAYER} tf.num_attention_heads=${TF_HEAD} tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL}

					done
				done
			done
		done
	done
done



# lr rate
python3 main.py model=tf batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=2 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=AVG lr=0.00001

python3 main.py model=tf batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=2 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=AVG lr=0.001

# glove
python3 main.py model=tf batch_size=128 embedding=glove sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=2 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=AVG
# comparable bert
python3 main.py model=tf batch_size=128 embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=2 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=AVG


tar cvfz experiments.tar.gz experiments/