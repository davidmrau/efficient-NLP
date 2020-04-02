
cd ..

BATCH_SIZE="10"
# Transformer parameters:
TF_LAYERS="2"
TF_HEADS="4"

TF_POOLS="AVG"
EMBEDDINGS="bert random glove"

L1_SCALARS="0"
# Dense
SPARSE_DIMENSIONS="1000"

TF_HID_DIM=500



for EMBEDDING in ${EMBEDDINGS}; do
	for SPARSE_DIMENSION in ${SPARSE_DIMENSIONS}; do
		for L1_SCALAR in ${L1_SCALARS}; do

			for TF_LAYER in ${TF_LAYERS}; do
				for TF_HEAD in ${TF_HEADS}; do
					for TF_POOL in ${TF_POOLS}; do

						# echo "Training"
						python3 main.py model=tf batch_size=${BATCH_SIZE} embedding=${EMBEDDING} sparse_dimensions=${SPARSE_DIMENSION} l1_scalar=${L1_SCALAR} \
						tf.num_of_layers=${TF_LAYER} tf.num_attention_heads=${TF_HEAD} tf.hidden_size=${TF_HID_DIM} tf.pooling_method=${TF_POOL} debug=True

					done
				done
			done
		done
	done
done



# random embeddings


# lr rate

# glove
