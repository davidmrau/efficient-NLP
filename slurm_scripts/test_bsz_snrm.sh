
MODEL=snrm

BATCH_SIZES="1024"

cd ..

# test what is the max batch size that can fit in the gpu. If the folloing trains at least one epoch, we are good for the rest experiments that follow
python3 main.py model=snrm batch_size=${BATCH_SIZE} embedding=bert sparse_dimensions=10000 l1_scalar=0 snrm.hidden_sizes=500-300-100-300-500 \
model_folder=Test_Batch_size_exp
