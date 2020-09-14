python3 inference.py model_folder=experiments_robust04/Dev...robust04_l1_0_margin_1_Emb_glove_Sparse_1000_bsz_128_lr_0.0001_SNRM_n-gram_cnn_100-300_provided_triplets_top_100 \
		rerank_top_N=100 \
		ranking_results=data/robust04/weak_overfitting/all/top_2000.csv \
		docs=data/robust04/robust04_raw_docs.num_query_glove_stop_lucene_remove_unk.tsv \
		queries=data/robust04/04.testset_num_query_lower_glove_stop_lucene_remove_unk.tsv metric=none \
		eval_1st_fold=False
