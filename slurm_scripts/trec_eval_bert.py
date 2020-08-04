MODEL_FOLDER=experiments_msmarco/Dev...msmarco_bsz_128_lr_1e-06_POINT_wise_BERT_loaded_experiments_lm.relevance_embedding_bert_orcas_mlm_p_0_2.checkpoint-38000_Layers_0-1-2-3-4-5-6-7-8-9-10-11-12/

python3 inference.py model_folder=$MODEL_FOLDER qrels=data/msmarco/2019qrels-pass.txt queries=msmarco-test2019-queries_43.tsv_bert_stop_none_remove_unk.tsv docs=collection.tsv_bert_stop_none_remove_unk.tsv ranking_results=data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv metric=map


