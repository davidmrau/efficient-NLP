

TRAIN_FILE=data/orcas/orcas-doctrain-qrels_train.tsv
TEST_FILE=data/orcas/orcas-doctrain-qrels_eval.tsv
DOCS_FILE=data/docs/msmarco-docs.tsv
QUERY_FILE=data/orcas/orcas-doctrain-queries.tsv
BATCH_SIZE=4

python3 run_language_modeling.py --model_type=bert --output_dir=experiments_lm/relevance_embedding_bert_orcas --train_data_file=$TRAIN_FILE --eval_data_file=$TEST_FILE --mlm --do_train --do_eval --per_gpu_train_batch_size=$BATCH_SIZE  --per_gpu_eval_batch_size=$BATCH_SIZE  --model_name_or_path=bert-base-uncased --ratio_first_second=0.8 --docs_path=$DOCS_FILE --query_path=$QUERY_FILE --mlm_probability=0.5 --exclude
