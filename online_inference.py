from dataset import Sequential
import torch

from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
from utils import collate_fn_padd_single, load_model
from torch.utils.data import DataLoader

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

	results_file_path = os.path.join(cfg.model_folder + '/ranking_results_online.' + cfg.query_file)
	results_file_path_trec = os.path.join(cfg.model_folder + '/ranking_results_online.' + cfg.query_file + '.trec')
	# Initialize an Inverted Index object
	ii = InvertedIndex(parent_dir=cfg.index_folder, vocab_size = cfg.sparse_dimensions, num_of_decimals=cfg.num_of_decimals)

	# open file
	dataset = Sequential(cfg.query_file, tokenize=cfg.tokenize, min_len=cfg.snrm.n)
	dataloader =  DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_padd_single)


	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)
	with torch.no_grad():
		model.eval()
		# open results file
		results_file = open(results_file_path, 'w')
		results_file_trec = open(results_file_path_trec, 'w')
		count = 0
		for batch_ids, batch_data, batch_lengths in dataloader:
			# print(batch_data)
			print(batch_ids)
			logits = model(batch_data.to(device), batch_lengths.to(device))
			results = ii.get_scores(batch_ids, logits.cpu(), top_results = cfg.top_results, max_candidates_per_posting_list = cfg.max_posting)
			if count % 1 == 0:
				print(count, ' batches processed')
			for query_id, result_list in results:
				for rank, (doc_id, score) in enumerate(result_list):
					results_file.write(f'{query_id}\t{doc_id}\t{rank + 1}\n' )
					results_file_trec.write(f'{query_id}\t0\t{doc_id}\t{rank+1}\t{score}\teval\n')

		results_file.close()
		results_file_trec.close()

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.query_file or not cl_cfg.batch_size or cl_cfg.tokenize == None or not cl_cfg.index_folder or not cl_cfg.max_posting:
		raise ValueError("usage: online_inference.py model_folder=MODEL_FOLDER index_folder=INDEX_FOLDER query_file=QUERY_FILE batch_size=BATCH_SIZE tokenize=True|False max_posting=-1")

	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	online_inference(cfg)
