
cd ..

DOCS_FILE=top1000.tsv.d_id_doc.tokenized.tsv
QUERY_FILE=msmarco-test2019-queries.tokenized.tsv
QRELS=qrels.train.tsv


EXPERIMENT_FOLDER='experiments/'

echo ${QRELS}
echo ${QUERY_FILE}
echo ${DOCS_FILE}
for HIDDEN in 100 200;do
	MODEL_FOLDER=${EXPERIMENT_FOLDER}${HIDDEN}_model
	MODEL_FOLDER='experiments/model_snrm_l1_scalar_1_lr_0.0001_drop_0.2_emb_bert_batch_size_64_debug_False/'
	echo ${MODEL_FOLDER} 
	python3 main.py model_folder=${MODEL_FOLDER}
	python3 create_index.py model_folder=${MODEL_FOLDER} docs_file=${DOCS_FILE}
	python3 online_inference.py model_folder=${MODEL_FOLDER} query_file=${QUERY_FILE} qrels=${QRELS}
done


