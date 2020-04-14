from dataset import MSMarcoSequential, MSMarcoSequentialDev
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
from utils import write_ranking, write_ranking_trec, write_pickle
import numpy as np
from ms_marco_eval import compute_metrics_from_files
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_data_loaders
from collections import defaultdict
# from inference import get_repr
matplotlib.use('Agg')

""" Run online inference (for test set) without inverted index
"""
# usage: online_inference.py model_path=model_path query_file=QUERY_FILE qrels=QRELS
# the script is loading FOLDER_TO_MODEL/best_model.model
#


def get_repr(model, dataloader, device):
	with torch.no_grad():
		model.eval()
		reprs = list()
		ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
			repr_ = model(batch_data_d.to(device), batch_lengths_d.to(device))
			reprs.append(repr_)
			ids += batch_ids_d
		return reprs, ids

def get_scores_excluding_dims(doc_reprs, doc_ids, q_reprs, top_results, exclude_dims = []):

	mask = torch.ones(q_reprs[0].size(1), device = q_reprs[0].device).float()
	for index in exclude_dims:
		mask[index] = 0

	bsz = q_reprs[0].size(0)

	batch_mask = mask.unsqueeze(dim = 0).repeat(bsz, 1)


	scores = list()
	for batch_q_repr in q_reprs:

		# repeat the mask for the whole batch of queries
		if batch_q_repr.size(0) == bsz:
			temp_mask = batch_mask
		else:
			temp_mask = mask.unsqueeze(dim = 0).repeat(batch_q_repr.size(0), 1)

		temp_batch_q_repr = batch_q_repr * temp_mask


		batch_len = len(batch_q_repr)
		# q_score_lists = [ []]*batch_len
		q_score_lists = [[] for i in range(batch_len) ]
		for batch_doc_repr in doc_reprs:

			dots_q_d = temp_batch_q_repr  @ batch_doc_repr.T
			# appending scores of batch_documents for this batch of queries
			for i in range(batch_len):
				q_score_lists[i] += dots_q_d[i].detach().cpu().tolist()


		# now we will sort the documents by relevance, for each query
		for i in range(batch_len):
			tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(doc_ids, q_score_lists[i])]
			sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse = True)
			if top_results != -1:
				sorted_by_relevance = sorted_by_relevance[:top_results]
			scores.append(sorted_by_relevance)

	return scores



def analysis(cfg, max_k = -1):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = torch.load(cfg.model_path + '/best_model.model', map_location=device)

	model = model.to(device=device)
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		if not isinstance(model, torch.nn.DataParallel):
			model = torch.nn.DataParallel(model)
	else:
		if isinstance(model, torch.nn.DataParallel):
			model = model.module





	metrics_file_path = cfg.model_path + f'/ranking_results.txt'
	metrics_file = open(metrics_file_path, 'w')

	qrels_base = cfg.qrels_val.split('/')[-1]

	ranking_file_path = f'{cfg.model_path}/{qrels_base}_re_ranking'



	top_results = 1000

	MaxMRRRank = 1000



	MRR_top_k_freq = []

	MRR_bottom_k_freq = []

	MRR_top_k_var = []

	MRR_bottom_k_var = []

	qrels = cfg.qrels_val


	# -----------------------
	with torch.no_grad():
		model.eval()
		# av_eval_loss, _ = run_epoch(model, dataloaders['val'], loss_fn, epoch, writer, l1_scalar, balance_scalar, total_training_steps, device)
		#run ms marco eval
		#

		# scores, q_repr, d_repr, q_ids, _ = evaluate_for_analysis(model, dataloaders['val'], device, top_results)

		# cfg.query_file_val.replace("//", "/")
		# cfg.docs_file_val.replace("//", "/")


		query_batch_generator = MSMarcoSequential(cfg.query_file_val, cfg.batch_size)
		docs_batch_generator = MSMarcoSequential(cfg.docs_file_val, cfg.batch_size)

		query_batch_generator.reset()
		docs_batch_generator.reset()


		d_repr, d_ids = get_repr(model, docs_batch_generator, device)
		q_repr, q_ids = get_repr(model, query_batch_generator, device)



		# get posting lists, without activations, and doc_id are represented by their index on the d_ids list
		posting_lists_doc_indeces= defaultdict(set)

		passed_samples_coutner = 0

		for doc_batch_repr in d_repr:
			for batch_index, doc_repr in enumerate(doc_batch_repr):
				nonzero_indeces = (doc_repr>0).nonzero().squeeze()

				for dimension in nonzero_indeces:
					posting_lists_doc_indeces[ dimension.item() ].add(batch_index + passed_samples_coutner)


			passed_samples_coutner += doc_batch_repr.size(0)







		#  rank dims wrt frequency (ignore dims that are always zeros)

		dim_frequency = torch.zeros(d_repr[0].size(1), device=device).float()

		dim_mean = torch.zeros(d_repr[0].size(1), device=device).float()
		number_of_docs = 0



		for i in range(len(d_repr)):
			# calculate frequency of each dimension
			dim_frequency  += (d_repr[i] > 0).sum(dim = 0).float()

			# calculate mean for each dimension 
			dim_mean += d_repr[i].sum(dim = 0)
			# calculate the number of total documents
			number_of_docs += d_repr[i].size(0)

		dim_mean /= number_of_docs


		# calcualte variance of each dimension 
		dim_var = torch.zeros(d_repr[0].size(1), device=device).float()

		for i in range(len(d_repr)):
			# calculate mean for each dimension 
			temp_dim_mean = dim_mean.unsqueeze(dim = 0).repeat(d_repr[i].size(0), 1)
			dim_var += ((d_repr[i] - temp_dim_mean)**2).sum(dim = 0)

		dim_var /= number_of_docs


		print(dim_frequency)


		number_of_used_dims = dim_frequency[ dim_frequency > 0].numel()

		# +1 so that at the end we can have results without any used dimensions and use it as "baseline"
		# this can be set as a parameter later
		if cfg.max_k == -1 :
			max_k = number_of_used_dims
		else:
			max_k = cfg.max_k

		print("Nubmer of used dimensions :", number_of_used_dims)


		sorted_dims_by_freq_ascending_order = dim_frequency.argsort().cpu().tolist()

		sorted_indeces_by_freq_ascending = sorted_dims_by_freq_ascending_order[ - number_of_used_dims:].copy()
		freq_ascending_total_docs_affected = []
		freq_ascending_last_dim_docs_affected = []

		sorted_indeces_by_freq_decending = sorted_indeces_by_freq_ascending.copy()
		sorted_indeces_by_freq_decending.reverse()
		freq_decending_total_docs_affected = []
		freq_decending_last_dim_docs_affected = []


		sorted_dims_by_var_ascending_order = dim_var.argsort().cpu().tolist()

		sorted_indeces_by_var_ascending = sorted_dims_by_var_ascending_order[ - number_of_used_dims:].copy()
		var_ascending_total_docs_affected = []
		var_ascending_last_dim_docs_affected = []

		sorted_indeces_by_var_decending = sorted_indeces_by_var_ascending.copy()
		sorted_indeces_by_var_decending.reverse()
		var_decending_total_docs_affected = []
		var_decending_last_dim_docs_affected = []

		# .argsort() returns indexes of elements in ascenting order. First index will be the one that points to the min element! if you want to reverse, set descending = True

		#  rank dims wrt variance (ignore dims that are always zeros)


		total_documents = len(d_ids)


		print("Most Frequent dimensions :")
		print(sorted_indeces_by_freq_decending[:max_k])
		print("Removing top k frequent, one by one")
		total_docs_affected = set()
		# top k frequent
		for k in range(max_k):



			tmp_ranking_file = f'{ranking_file_path}_tmp'

			scores = get_scores_excluding_dims(d_repr, d_ids, q_repr, top_results, exclude_dims = sorted_indeces_by_freq_decending[:k])

			write_ranking(scores, q_ids, tmp_ranking_file, MaxMRRRank)

			metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = tmp_ranking_file, MaxMRRRank=MaxMRRRank)
			MRR = metrics[f'MRR @{MaxMRRRank}']

			MRR_top_k_freq.append(MRR)

			# writer.add_scalar(f'Eval_MRR@1000', MRR, total_training_steps  )
			print(f'{k} -  MRR@1000: {MRR}')


			doc_indeces_affected_by_this_dimension = posting_lists_doc_indeces[sorted_indeces_by_freq_decending[k]]

			total_docs_affected = total_docs_affected.union(doc_indeces_affected_by_this_dimension)


			freq_decending_total_docs_affected.append(len(total_docs_affected) / float(total_documents) )
			freq_decending_last_dim_docs_affected.append(len(doc_indeces_affected_by_this_dimension) / float(total_documents) )


		print("Least Frequent dimensions :")
		print(sorted_indeces_by_freq_ascending[:max_k])
		print("Removing bottom k frequent, one by one")
		total_docs_affected = set()
		# top k frequent
		for k in range(max_k):

			tmp_ranking_file = f'{ranking_file_path}_tmp'

			scores = get_scores_excluding_dims(d_repr, d_ids, q_repr, top_results, exclude_dims = sorted_indeces_by_freq_ascending[:k])

			write_ranking(scores, q_ids, tmp_ranking_file, MaxMRRRank)

			metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = tmp_ranking_file, MaxMRRRank=MaxMRRRank)
			MRR = metrics[f'MRR @{MaxMRRRank}']

			MRR_bottom_k_freq.append(MRR)

			# writer.add_scalar(f'Eval_MRR@1000', MRR, total_training_steps  )
			print(f'{k} -  MRR@1000: {MRR}')

			doc_indeces_affected_by_this_dimension = posting_lists_doc_indeces[sorted_indeces_by_freq_ascending[k]]

			total_docs_affected = total_docs_affected.union(doc_indeces_affected_by_this_dimension)

			freq_ascending_total_docs_affected.append( len(total_docs_affected) / float(total_documents))
			freq_ascending_last_dim_docs_affected.append(len(doc_indeces_affected_by_this_dimension) / float(total_documents) )


		print("Most Variant dimensions :")
		print(sorted_indeces_by_var_decending[:max_k])
		print("Removing top k variant, one by one")
		total_docs_affected = set()
		# top k frequent
		for k in range(max_k):

			tmp_ranking_file = f'{ranking_file_path}_tmp'

			scores = get_scores_excluding_dims(d_repr, d_ids, q_repr, top_results, exclude_dims = sorted_indeces_by_var_decending[:k])

			write_ranking(scores, q_ids, tmp_ranking_file, MaxMRRRank)

			metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = tmp_ranking_file, MaxMRRRank=MaxMRRRank)
			MRR = metrics[f'MRR @{MaxMRRRank}']

			MRR_top_k_var.append(MRR)

			# writer.add_scalar(f'Eval_MRR@1000', MRR, total_training_steps  )
			print(f'{k} -  MRR@1000: {MRR}')

			doc_indeces_affected_by_this_dimension = posting_lists_doc_indeces[sorted_indeces_by_var_decending[k]]

			total_docs_affected = total_docs_affected.union(doc_indeces_affected_by_this_dimension)


			var_decending_total_docs_affected.append( len(total_docs_affected) / float(total_documents))
			var_decending_last_dim_docs_affected.append(len(doc_indeces_affected_by_this_dimension) / float(total_documents) )


		print("Least Variant dimensions :")
		print(sorted_indeces_by_var_ascending[:max_k])
		print("Removing bottom k variant, one by one")
		total_docs_affected = set()
		# top k frequent
		for k in range(max_k):

			tmp_ranking_file = f'{ranking_file_path}_tmp'

			scores = get_scores_excluding_dims(d_repr, d_ids, q_repr, top_results, exclude_dims = sorted_indeces_by_var_ascending[:k])

			write_ranking(scores, q_ids, tmp_ranking_file, MaxMRRRank)

			metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = tmp_ranking_file, MaxMRRRank=MaxMRRRank)
			MRR = metrics[f'MRR @{MaxMRRRank}']

			MRR_bottom_k_var.append(MRR)

			# writer.add_scalar(f'Eval_MRR@1000', MRR, total_training_steps  )
			print(f'{k} -  MRR@1000: {MRR}')

			doc_indeces_affected_by_this_dimension = posting_lists_doc_indeces[sorted_indeces_by_var_ascending[k]]

			total_docs_affected = total_docs_affected.union(doc_indeces_affected_by_this_dimension)

			var_ascending_total_docs_affected.append( len(total_docs_affected) / float(total_documents))
			var_ascending_last_dim_docs_affected.append(len(doc_indeces_affected_by_this_dimension) / float(total_documents) )



	top_k_analysis_dict = {}
	top_k_analysis_dict["MRR_top_k_freq"] = MRR_top_k_freq
	top_k_analysis_dict["MRR_bottom_k_freq"] = MRR_bottom_k_freq
	top_k_analysis_dict["MRR_top_k_var"] = MRR_top_k_var
	top_k_analysis_dict["MRR_bottom_k_var"] = MRR_bottom_k_var


	top_k_analysis_dict["most_freq_dims"] = sorted_indeces_by_freq_decending[:max_k]
	top_k_analysis_dict["least_freq_dims"] = sorted_indeces_by_freq_ascending[:max_k]
	top_k_analysis_dict["most_var_dims"] = sorted_indeces_by_var_decending[:max_k]
	top_k_analysis_dict["least_var_dims"] = sorted_indeces_by_var_ascending[:max_k]


	top_k_analysis_dict["top_k_freq_camulative_counter"] = freq_decending_total_docs_affected
	top_k_analysis_dict["top_k_freq_step_counter"] = freq_decending_last_dim_docs_affected
	top_k_analysis_dict["bottom_k_freq_camulative_counter"] = freq_ascending_total_docs_affected
	top_k_analysis_dict["bottom_k_freq_step_counter"] = freq_ascending_last_dim_docs_affected

	top_k_analysis_dict["top_k_var_camulative_counter"] = var_decending_total_docs_affected
	top_k_analysis_dict["top_k_var_step_counter"] = var_decending_last_dim_docs_affected


	top_k_analysis_dict["bottom_k_var_camulative_counter"] = var_ascending_total_docs_affected
	top_k_analysis_dict["bottom_k_var_step_counter"] = var_ascending_last_dim_docs_affected


	

	
	for k in top_k_analysis_dict:
		print(k)
		print(top_k_analysis_dict[k])









	write_pickle(top_k_analysis_dict, cfg.model_path+"/top_k_analysis_dict.pickle")



if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_path :
		raise ValueError("usage: analysis.py model_path=MODEL_PATH")
	if not cl_cfg.max_k :
		cl_cfg.max_k = -1
		print("max_k parameter was not set. Applying analysis for all used dimensions")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_path}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	analysis(cfg)
