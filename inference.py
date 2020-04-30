from dataset import MSMarcoSequential, MSMarcoSequentialDev
import torch
import pickle
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf
import os
from utils import write_ranking, write_ranking_trec, l0_loss, instantiate_model
import numpy as np
from ms_marco_eval import compute_metrics_from_files

from run_model import evaluate
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from snrm import SNRM
from bert_based import BERT_based
matplotlib.use('Agg')

""" Run online inference (for test set) without inverted index
"""



def load_model(cfg, device):
	
	model_old = torch.load(cfg.model_path + '/best_model.model', map_location=device)
	
	if isinstance(model_old, torch.nn.DataParallel):
		model_old = model_old.module

	state_dict = model_old.state_dict()

	model, _ = instantiate_model(cfg)
	
	model.load_state_dict(state_dict)

	return model


def inference(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	model = load_model(cfg, device)

	metrics_file_path = cfg.model_path + f'/ranking_results.txt'
	metrics_file = open(metrics_file_path, 'w')

	qrels_base = cfg.qrels.split('/')[-1]

	ranking_file_path = f'{cfg.model_path}/{qrels_base}_re_ranking'

	# set minimum length in case of snrm
	min_len = 0
	if cfg.model == "snrm":
		min_len = cfg.snrm.n

	# load data
	query_batch_generator = MSMarcoSequentialDev(cfg.q_docs, cfg.batch_size, cfg.glove_word2idx_path, embedding=cfg.embedding, is_query=True,
									min_len = min_len, max_len=cfg.max_input_len)
	docs_batch_generator = MSMarcoSequentialDev(cfg.q_docs, cfg.batch_size, cfg.glove_word2idx_path,embedding=cfg.embedding, is_query=False,
									min_len = min_len, max_len=cfg.max_input_len)

	av_l0_query = list()
	av_l0_doc = list()
	
	data_loaders = [query_batch_generator.reset(), docs_batch_generator.reset()]
	scores, q_repr, d_repr, q_ids_q, d_ids_q = evaluate(model, data_loaders, device, cfg.top_results, reset=False)
	q_ids = q_ids_q
	av_l0_query.append(l0_loss(torch.cat(q_repr)))
	av_l0_doc.append(l0_loss(torch.cat(d_repr)))
	while len(d_repr) > 0:
		print(q_ids_q)
		scores_q, q_repr, d_repr, q_ids_q, d_ids_q = evaluate(model, data_loaders, device, cfg.top_results, reset=False)
		scores += scores_q
		q_ids += q_ids_q
		if len(d_repr) > 0:
			av_l0_query.append(l0_loss(torch.cat(q_repr)))
			av_l0_doc.append(l0_loss(torch.cat(d_repr)))
	
	write_ranking(scores, q_ids, ranking_file_path, cfg.MaxMRRRank)
	write_ranking_trec(scores, q_ids, ranking_file_path+'.trec', cfg.MaxMRRRank)

	metrics_file.write(f'{qrels_base} dim used docs:\t{int(torch.stack(av_l0_doc).mean().item() * cfg.sparse_dimensions)}\n')
	metrics_file.write(f'{qrels_base} dim used query:\t{int(torch.stack(av_l0_query).mean().item() * cfg.sparse_dimensions)}\n')

	metrics_file.write(f'{qrels_base} l0 docs:\t{round(torch.stack(av_l0_query).mean().item(), 5)}\n')
	metrics_file.write(f'{qrels_base} l0 query:\t{round(torch.stack(av_l0_doc).mean().item(), 5)}\n')
	metric_score = compute_metrics_from_files(path_to_reference = cfg.qrels, path_to_candidate = ranking_file_path, MaxMRRRank=cfg.MaxMRRRank)

	# returning the MRR @ 1000
	metrics_file.write(f'{qrels_base} MRR@{cfg.MaxMRRRank}:\t{metric_score}\n')
	metrics_file.close()

	print(f'{qrels_base} MRR@{cfg.MaxMRRRank}:\t{metric_score}\n')
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
	cfg_default = OmegaConf.load('config.yaml')
	cfg = OmegaConf.merge(cfg_default, cfg)
	inference(cfg)
