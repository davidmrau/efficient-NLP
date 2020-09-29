EXPERIMENTS_DIR=/project/draugpu/models/msmarco_bert_interaction_based/

MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2/
# MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3/
# MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3-4-5-6/
# MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3-4-5-6-7-8-9/
# MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3-4-5-6-7-8-9-10-11-12/

QUERIES=data/msmarco/msmarco-test2020-queries.tsv_bert_stop_none_remove_unk.tsv
DOCS=data/msmarco/collection.tsv_bert_stop_none_remove_unk.tsv
# RANKING_RESULTS=data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style.tsv
RANKING_RESULTS=data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_only_54_judged.tsv
# RANKING_RESULTS=data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style_only_1.tsv


METRIC=map


FINAL_MODEL_PATH="${EXPERIMENTS_DIR}${MODEL_FOLDER}"

# Qrels for all metrics except for NDCG
# QRELS=data/msmarco/2020-qrels-pass-no1.txt
# Qrels for NDCG Metrics
QRELS=data/msmarco/2020-qrels-pass-final.txt

python3 inference.py model_folder=$FINAL_MODEL_PATH queries=$QUERIES docs=$DOCS ranking_results=$RANKING_RESULTS  metric=$METRIC qrels=$QRELS rerank_top_N=-1 eval_1st_fold=False


MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3/
FINAL_MODEL_PATH="${EXPERIMENTS_DIR}${MODEL_FOLDER}"
python3 inference.py model_folder=$FINAL_MODEL_PATH queries=$QUERIES docs=$DOCS ranking_results=$RANKING_RESULTS  metric=$METRIC qrels=$QRELS rerank_top_N=-1 eval_1st_fold=False


MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3-4-5-6/
FINAL_MODEL_PATH="${EXPERIMENTS_DIR}${MODEL_FOLDER}"
python3 inference.py model_folder=$FINAL_MODEL_PATH queries=$QUERIES docs=$DOCS ranking_results=$RANKING_RESULTS  metric=$METRIC qrels=$QRELS rerank_top_N=-1 eval_1st_fold=False


MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3-4-5-6-7-8-9/
FINAL_MODEL_PATH="${EXPERIMENTS_DIR}${MODEL_FOLDER}"
python3 inference.py model_folder=$FINAL_MODEL_PATH queries=$QUERIES docs=$DOCS ranking_results=$RANKING_RESULTS  metric=$METRIC qrels=$QRELS rerank_top_N=-1 eval_1st_fold=False

MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2-3-4-5-6-7-8-9-10-11-12/
FINAL_MODEL_PATH="${EXPERIMENTS_DIR}${MODEL_FOLDER}"
python3 inference.py model_folder=$FINAL_MODEL_PATH queries=$QUERIES docs=$DOCS ranking_results=$RANKING_RESULTS  metric=$METRIC qrels=$QRELS rerank_top_N=-1 eval_1st_fold=False
