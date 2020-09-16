EXPERIMENTS_DIR=/project/draugpu/models/msmarco_bert_interaction_based/

MODEL_FOLDER=msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_default_Layers_0-1-2/

python3 inference.py model_folder=$EXPERIMENTS_DIR$MODEL_FOLDER queries=data/msmarco/msmarco-test2020-queries.tsv_bert_stop_none_remove_unk.tsv docs=data/msmarco/collection.tsv_bert_stop_none_remove_unk.tsv ranking_results=data/msmarco/msmarco-passagetest2020-top1000_ranking_results_style.tsv  metric=none



