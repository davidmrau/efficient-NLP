from dataset import MSMarcoSequential, MSMarcoSequentialDev
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
from utils import write_ranking, write_ranking_trec
import numpy as np
from ms_marco_eval import compute_metrics_from_files
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
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


def evaluate(model, data_loaders, device, top_results, reset=True):

	query_batch_generator, docs_batch_generator = data_loaders

	if reset:
		docs_batch_generator.reset()
		query_batch_generator.reset()

	d_repr, d_ids = get_repr(model, docs_batch_generator, device)
	q_repr, q_ids = get_repr(model, query_batch_generator, device)
	scores = get_scores(d_repr, d_ids, q_repr, top_results)
	# returning the MRR @ 1000
	return scores, q_repr, d_repr, q_ids, d_ids


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

	model = torch.load(cfg.model_path + '/best_model.model', map_location=device)

	model = model.to(device=device)
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = torch.nn.DataParallel(model)

	metrics_file_path = cfg.model_path + f'/ranking_results.txt'
	metrics_file = open(metrics_file_path, 'w')

	qrels_base = cfg.qrels.split('/')[-1]

	ranking_file_path = f'{cfg.model_path}/{qrels_base}_re_ranking'

	# load data
	#
	# query_batch_generator = MSMarcoSequential(cfg.query_file_val, cfg.batch_size)
	# docs_batch_generator = MSMarcoSequential(cfg.docs_file_val, cfg.batch_size)

	query_batch_generator = MSMarcoSequentialDev(cfg.q_docs, cfg.batch_size, cfg.glove_word2idx_path, embedding=cfg.embedding, is_query=True)
	docs_batch_generator = MSMarcoSequentialDev(cfg.q_docs, cfg.batch_size, cfg.glove_word2idx_path,embedding=cfg.embedding, is_query=False)

	data_loaders = [query_batch_generator.reset(), docs_batch_generator.reset()]
	scores, q_repr, d_repr, q_ids_q, d_ids_q = evaluate(model, data_loaders, device, cfg.top_results, reset=False)
	q_ids = q_ids_q
	while len(d_repr) > 0:
		print(q_ids_q)
		scores_q, q_repr, d_repr, q_ids_q, d_ids_q = evaluate(model, data_loaders, device, cfg.top_results, reset=False)
		scores += scores_q
		q_ids += q_ids_q


	write_ranking(scores, q_ids, ranking_file_path, cfg.MaxMRRRank)
	write_ranking_trec(scores, q_ids, ranking_file_path+'.trec', cfg.MaxMRRRank)


	metric = compute_metrics_from_files(path_to_reference = cfg.qrels, path_to_candidate = ranking_file_path, MaxMRRRank=cfg.MaxMRRRank)

	# returning the MRR @ 1000
	metrics_file.write(f'{qrels_base} MRR@{cfg.MaxMRRRank}:\t{metric}\n')
	metrics_file.close()

	print(f'{qrels_base} MRR@{cfg.MaxMRRRank}:\t{metric}\n')
if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_path or not cl_cfg.MaxMRRRank or not cl_cfg.q_docs or not cl_cfg.qrels :
		raise ValueError("usage: inference.py model_path=MODEL_PATH qrels=QRELS_PATH q_docs=QUERY_DOCS_PATH MaxMRRRank=100 ")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_path}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	inference(cfg)
