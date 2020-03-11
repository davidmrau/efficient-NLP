from dataset import MSMarcoSequential
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
import numpy as np
from ms_marco_eval import compute_metrics_from_files

""" Run online inference (for test set) without inverted index
"""
# usage: online_inference.py model_folder=MODEL_FOLDER query_file=QUERY_FILE qrels=QRELS
# the script is loading FOLDER_TO_MODEL/best_model.model
#

def inference(cfg, model = None):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	top_results = cfg.top_results


	query_batch_generator = MSMarcoSequential(cfg.dataset_path + cfg.query_file, cfg.batch_size).batch_generator()
	docs_batch_generator = MSMarcoSequential(cfg.dataset_path + cfg.docs_file, cfg.batch_size).batch_generator()

	metrics_file_path = cfg.model_folder + '/metrics.' + cfg.query_file
	results_file_path = cfg.model_folder + '/ranking_results.' + cfg.query_file
	doc_reprs_file_path = cfg.model_folder + '/doc_reprs.' + cfg.docs_file

	metrics_file = open(metrics_file_path, 'w')
	results_file = open(results_file_path, 'w')
	results_file_trec = open(results_file_path+ '.trec', 'w')


	if model is None:
		model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)
		
	with torch.no_grad():
		model.eval()
		# open results file

		count = 0
		doc_reprs = list()
		doc_ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in docs_batch_generator:
			doc_repr = model(batch_data_d.to(device), batch_lengths_d.to(device))
			# print(doc_repr[:10,:10])
			# exit()
			doc_reprs.append(doc_repr.T)
			doc_ids += batch_ids_d
			if count % 100 == 0:
				print(count, ' batches processed')



			count += 1
		print('docs forward pass done!')

		pickle.dump([doc_ids, doc_reprs], open(doc_reprs_file_path + '.p', 'wb'))

		# save logits to file
		count = 0
		for batch_ids_q, batch_data_q, batch_lengths_q in query_batch_generator:
			batch_len = len(batch_ids_q)
			q_reprs = model(batch_data_q.to(device), batch_lengths_q.to(device))
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
			if count % 1 == 0:
				print(count, ' batches processed')

			count += 1
		results_file.close()
		results_file_trec.close()
		metrics = compute_metrics_from_files(path_to_reference = cfg.dataset_path + cfg.qrels, path_to_candidate = results_file_path)


		for metric in metrics:
			metrics_file.write(f'{metric}:\t{metrics[metric]}\n')

		metrics_file.close()

		# returning the MRR @ 1000
		return metrics['MRR @1000']

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.query_file or not cl_cfg.qrels or not cl_cfg.docs_file:
		raise ValueError("usage: inference.py model_folder=MODEL_FOLDER docs_file=DOCS_FILE query_file=QUERY_FILE qrels=QRELS top_results=1000")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	inference(cfg)
