from dataset import MSMarcoSequential
import torch

from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
from ms_marco_eval import compute_metrics_from_files

""" Load an pre-built inverted index and run online inference (for test set)
"""
# usage: online_inference.py model_folder=MODEL_FOLDER query_file=QUERY_FILE qrels=QRELS
# the script is loading FOLDER_TO_MODEL/best_model.model
#


def online_inference(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	results_file_path = os.path.join(cfg.model_folder + '/ranking_results.' + cfg.query_file)
	'''
	# Initialize an Inverted Index object
	ii = InvertedIndex(parent_dir=cfg.model_folder, vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)

	ms_batch_generator = MSMarcoSequential(cfg.dataset_path + cfg.query_file, cfg.batch_size).batch_generator()

	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)

	# open results file
	results_file_path = os.path.join(cfg.model_folder + '/ranking_results.' + cfg.query_file)
	results_file = open(results_file_path, 'w')
	count = 0
	for batch_ids, batch_data, batch_lengths in ms_batch_generator:
		# print(batch_data)
		logits = model(batch_data.to(device), batch_lengths.to(device))
		results = ii.get_scores(batch_ids, logits.cpu(), top_results = 10, max_candidates_per_posting_list = 1000)
		if count % 1 == 0:
			print(count, ' batches processed')
		for query_id, result_list in results:
			for rank, (doc_id, score) in enumerate(result_list):
				results_file.write(f'{query_id}\t{doc_id}\t{rank + 1}\n' )

	results_file.close()
	'''
	metrics = compute_metrics_from_files(path_to_reference = cfg.dataset_path + cfg.qrels, path_to_candidate = results_file_path)

	metrics_file = open(os.path.join(cfg.model_folder + '/metrics.' + cfg.query_file, w))

	for metric in metrics:
		metrics_file_path.write(f'{metric}:\t{metrics[metric]}\n')

	metrics_file.close()

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.query_file or not cl_cfg.qrels:
		raise ValueError("usage: online_inference.py model_folder=MODEL_FOLDER query_file=QUERY_FILE qrels=QRELS")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	online_inference(cfg)
