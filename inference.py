from dataset import MSMarcoSequential
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

def evaluate(model, data_loaders, model_folder, qrels, dataset_path, sparse_dimensions, top_results, device, MaxMRRRank=1000):

	query_batch_generator, docs_batch_generator = data_loaders
	results_file_path = model_folder + f'/ranking_results_MRRRank_{MaxMRRRank}'
	doc_reprs_file_path = model_folder + '/doc_reprs'

	results_file = open(results_file_path, 'w')
	results_file_trec = open(results_file_path+ '.trec', 'w')

	with torch.no_grad():
		model.eval()
		# open results file

		doc_reprs = list()
		doc_ids = list()
		posting_lengths = np.zeros(sparse_dimensions)
		latent_terms_per_doc = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in docs_batch_generator.batch_generator():
			doc_repr = model(batch_data_d.to(device), batch_lengths_d.to(device))

			posting_lengths += (doc_repr > 0).sum(0).detach().cpu().numpy()
			latent_terms_per_doc += list((doc_repr > 0).sum(1).detach().cpu().numpy())
			
			doc_reprs.append(doc_repr.T)
			doc_ids += batch_ids_d
			
		plot_ordered_posting_lists_lengths(model_folder, posting_lengths, 'docs')
		plot_histogram_of_latent_terms(model_folder, latent_terms_per_doc, sparse_dimensions, 'docs')
		#pickle.dump([doc_ids, doc_reprs], open(doc_reprs_file_path + '.p', 'wb'))

		# save logits to file
		posting_lengths = np.zeros(sparse_dimensions)
		latent_terms_per_q = list()

		for batch_ids_q, batch_data_q, batch_lengths_q in query_batch_generator.batch_generator():
			batch_len = len(batch_ids_q)
			q_reprs = model(batch_data_q.to(device), batch_lengths_q.to(device))

			posting_lengths += (q_reprs > 0).sum(0).detach().cpu().numpy()
			latent_terms_per_q += list((q_reprs > 0).sum(1).detach().cpu().numpy())

			# q_score_lists = [ []]*batch_len
			q_score_lists = [[] for i in range(batch_len) ]
			for doc_repr in doc_reprs:
				dots_q_d = q_reprs @ doc_repr
				# appending scores of batch_documents for this batch of queries
				for i in range(batch_len):
					q_score_lists[i] += dots_q_d[i].detach().cpu().tolist()

			# at this point, we have a list of document scores, for each query in the batch

			# now we will sort the documents by relevance, for each query
			for i in range(batch_len):
				tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(doc_ids, q_score_lists[i])]
				sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse = True)
				query_id = batch_ids_q[i]
				if top_results != -1:
					sorted_by_relevance = sorted_by_relevance[:top_results]
				for j, (doc_id, score) in enumerate(sorted_by_relevance):
					results_file.write(f'{query_id}\t{doc_id}\t{j+1}\n' )
					results_file_trec.write(f'{query_id}\t0\t{doc_id}\t{j+1}\t{score}\teval\n')


		plot_ordered_posting_lists_lengths(model_folder, posting_lengths, 'query')
		plot_histogram_of_latent_terms(model_folder, latent_terms_per_q, sparse_dimensions, 'query')

		results_file.close()
		results_file_trec.close()
		metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = results_file_path, MaxMRRRank=MaxMRRRank)

		# returning the MRR @ 1000
		return metrics[f'MRR @{MaxMRRRank}']


def inference(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)

	metrics_file_path = cfg.model_folder + f'/eval.txt'
	metrics_file = open(metrics_file_path, 'w')
	# load data
	query_batch_generator = MSMarcoSequential(cfg.query_file_val, cfg.batch_size)
	docs_batch_generator = MSMarcoSequential(cfg.docs_file_val, cfg.batch_size)

	dataloader = [query_batch_generator, docs_batch_generator]
	top_results = cfg.top_results
	metric = evaluate(model, dataloader, cfg.model_folder, cfg.qrels_val, cfg.dataset_path, cfg.sparse_dimensions, cfg.top_results, device, MaxMRRRank=cfg.MaxMRRRank)

	metrics_file.write(f'MRR@{cfg.MaxMRRRank}:\t{metric}\n')

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
