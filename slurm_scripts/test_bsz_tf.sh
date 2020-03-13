

MODEL=tf

BATCH_SIZE="512"

cd ..

# test what is the max batch size that can fit in the gpu. If the folloing trains at least one epoch, we are good for the rest experiments that follow
python3 main.py model=${MODEL} batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 \
tf.num_of_layers=8 tf.num_attention_heads=8 tf.hidden_size=768 tf.pooling_method=CLS  \
model_folder=Test_Batch_size_exp num_epochs=1
