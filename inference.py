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


def get_repr(model, dataloader):
	with torch.no_grad():
		model.eval()
		reprs = list()
		ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
			doc_repr = model(batch_data_d.to(device), batch_lengths_d.to(device))

			reprs.append(repr.T)
			ids += batch_ids_d
		return reprs, ids

def get_scores(doc_reprs, doc_ids, q_reprs, q_ids, top_results):

	scores = {}
	for i in range(len(q_ids):
		q_score = list()
		q_repr = q_reprs[i]
		q_id = q_ids[i]
		for j in range(len(doc_ids)):
			score = q_repr @ doc_repr[j]
			doc_id = doc_ids[j]
			q_score.append((doc_id, score))

		q_score = sorted(q_score, key=lambda x: x[1], reverse = True)
		if top_results != -1:
			q_score = q_score[:top_results]
		scores[q_id] = q_score

	return scores

def write_scores(scores, model_folder, MaxMRRRank):
	results_file_path = model_folder + f'/ranking_results_MRRRank_{MaxMRRRank}'
	doc_reprs_file_path = model_folder + '/doc_reprs'

	results_file = open(results_file_path, 'w')
	results_file_trec = open(results_file_path+ '.trec', 'w')
	for q_id in scores:
		for j, (doc_id, score) in enumerate(scores[q_id]):
			results_file.write(f'{q_id}\t{doc_id}\t{j+1}\n' )
			results_file_trec.write(f'{q_id}\t0\t{doc_id}\t{j+1}\t{score}\teval\n')

	results_file.close()
	results_file_trec.close()

def evaluate(model, data_loaders, model_folder, qrels, dataset_path, sparse_dimensions, top_results, device, MaxMRRRank=1000):

	query_batch_generator, docs_batch_generator = data_loaders
	d_repr, d_ids = get_repr(model, docs_batch_generator)

	plot_ordered_posting_lists_lengths(model_folder, d_repr, 'docs')
	plot_histogram_of_latent_terms(model_folder, d_repr, sparse_dimensions, 'docs')

	q_repr, q_ids = get_repr(model, query_batch_generator)

	plot_ordered_posting_lists_lengths(model_folder, q_repr, 'query')
	plot_histogram_of_latent_terms(model_folder, q_repr, sparse_dimensions, 'query')

	scores = get_scores(d_repr, d_ids, q_repr, q_ids, top_results)

	write_scores(scores, model_folder, MaxMRRRank)
	metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = results_file_path, MaxMRRRank=MaxMRRRank)

	# returning the MRR @ 1000
	return metrics[f'MRR @{MaxMRRRank}']


def evaluate_dev():
	while:
		dataloder(dev=True):
		repr_d, ids = get_repr(model, dataloader)
		plot(repr_d)
		repr_q, ids = get_repr(model, dataloader)
		plot(repr_d)
		scores = score(repr_d, d_ids, repr_q)
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
