
cd ..

DOCS_FILE=top1000.tsv.d_id_doc.tokenized.tsv
QUERY_FILE=msmarco-test2019-queries.tokenized.tsv
QRELS=qrels.dev.tsv

python3 main.py debug=True
echo ${QRELS}
echo ${QUERY_FILE}
echo ${DOCS_FILE}
echo ${MODEL_FOLDER}

#python3 create_index.py model_folder=${MODEL_FOLDER} docs_file=${DOCS_FILE}
#python3 online_inference.py model_folder=${MODEL_FOLDER} query_file=${QUERY_FILE} qrels=${QRELS}



