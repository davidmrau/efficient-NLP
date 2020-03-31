from dataset import MSMarcoSequential, MSMarcoSequentialDev
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
from utils import plot_histogram_of_latent_terms, plot_ordered_posting_lists_lengths
import numpy as np
from ms_marco_eval import compute_metrics_from_files
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')

""" Run online inference (for test set) without inverted index
"""
# usage: online_inference.py model_folder=MODEL_FOLDER query_file=QUERY_FILE qrels=QRELS
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

def get_scores(doc_reprs, doc_ids, q_reprs, top_results):
	scores = list()
	for batch_q_repr in q_reprs:
		batch_len = len(batch_q_repr)
		# q_score_lists = [ []]*batch_len
		q_score_lists = [[] for i in range(batch_len) ]
		for batch_doc_repr in doc_reprs:
			dots_q_d = batch_q_repr @ batch_doc_repr.T
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
	#print(scores[0])
	return scores

def write_scores(scores, q_ids, results_file_path, MaxMRRRank):

	results_file = open(results_file_path, 'w')
	results_file_trec = open(results_file_path+ '.trec', 'w')
	for i, q_id in enumerate(q_ids):
		for j, (doc_id, score) in enumerate(scores[i]):
			results_file.write(f'{q_id}\t{doc_id}\t{j+1}\n' )
			results_file_trec.write(f'{q_id}\t0\t{doc_id}\t{j+1}\t{score}\teval\n')

	results_file.close()
	results_file_trec.close()

def evaluate(model, data_loaders, model_folder, qrels, dataset_path, sparse_dimensions, top_results, device, MaxMRRRank=1000):

	results_file_path = model_folder + f'/ranking_results_MRRRank_{MaxMRRRank}'

	query_batch_generator, docs_batch_generator = data_loaders
	docs_batch_generator.reset()
	d_repr, d_ids = get_repr(model, docs_batch_generator, device)

	plot_ordered_posting_lists_lengths(model_folder, d_repr, 'docs')
	plot_histogram_of_latent_terms(model_folder, d_repr, sparse_dimensions, 'docs')

	query_batch_generator.reset()
	q_repr, q_ids = get_repr(model, query_batch_generator, device)

	plot_ordered_posting_lists_lengths(model_folder, q_repr, 'query')
	plot_histogram_of_latent_terms(model_folder, q_repr, sparse_dimensions, 'query')

	scores = get_scores(d_repr, d_ids, q_repr, top_results)

	write_scores(scores, q_ids, results_file_path, MaxMRRRank)
	metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = results_file_path, MaxMRRRank=MaxMRRRank)

	# returning the MRR @ 1000
	return metrics[f'MRR @{MaxMRRRank}']


#def evaluate_dev():
	#while True:
#		dataloder(dev=True):
#		repr_d, ids = get_repr(model, dataloader)
#		plot(repr_d)
#		repr_q, ids = get_repr(model, dataloader)
#		plot(repr_d)
#		scores = score(repr_d, d_ids, repr_q)
	#restuls write and aggreate


def inference(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)

	metrics_file_path = cfg.model_folder + f'/eval.txt'
	metrics_file = open(metrics_file_path, 'w')
	# load data
	#
	# query_batch_generator = MSMarcoSequential(cfg.query_file_val, cfg.batch_size)
	# docs_batch_generator = MSMarcoSequential(cfg.docs_file_val, cfg.batch_size)

	query_batch_generator = MSMarcoSequentialDev(cfg.q_docs_file_val, cfg.batch_size, cfg.glove_word2idx_path, embedding=cfg.embedding, is_query=True)
	docs_batch_generator = MSMarcoSequentialDev(cfg.q_docs_file_val, cfg.batch_size, cfg.glove_word2idx_path,embedding=cfg.embedding, is_query=False)


	dataloader = [query_batch_generator, docs_batch_generator]
	top_results = cfg.top_results

#	metric = evaluate(model, dataloader, cfg.model_folder, cfg.qrels_dev, cfg.dataset_path, cfg.sparse_dimensions, cfg.top_results, device, MaxMRRRank=cfg.MaxMRRRank)

	results_file_path = cfg.model_folder + f'/ranking_results_MRRRank_{cfg.MaxMRRRank}_dev'
	
	docs_batch_generator.reset()
	query_batch_generator.reset()
	
	d_repr, d_ids = get_repr(model, docs_batch_generator, device)
	q_repr, q_ids_q = get_repr(model, query_batch_generator, device)
	scores= get_scores(d_repr, d_ids, q_repr, top_results)

	q_ids = q_ids_q

	while len(d_repr) > 0:
		d_repr, d_ids = get_repr(model, docs_batch_generator, device)
		q_repr, q_ids_q = get_repr(model, query_batch_generator, device)

		scores += get_scores(d_repr, d_ids, q_repr, top_results)
		q_ids += q_ids_q

	write_scores(scores, q_ids, results_file_path, cfg.MaxMRRRank)

	

	metric = compute_metrics_from_files(path_to_reference = cfg.qrels_val, path_to_candidate = results_file_path, MaxMRRRank=cfg.MaxMRRRank)

	# returning the MRR @ 1000
	metrics_file.write(f'MRR@{cfg.MaxMRRRank} dev:\t{metric}\n')

	metrics_file.close()

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.MaxMRRRank:
		raise ValueError("usage: inference.py model_folder=MODEL_FOLDER MaxMRRRank=100")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	inference(cfg)
