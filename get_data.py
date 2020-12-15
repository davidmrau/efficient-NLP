
import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use('Agg')
import os
import pickle

from enlp.utils import load_model, instantiate_model
from enlp.file_interface import File, FileInterface
from enlp.dataset import SingleSequential, collate_fn_bert_interaction_single
from torch.utils.data import DataLoader
from enlp.get_reprs import reprs_bert_interaction


""" Run online inference (for test set) without inverted index
"""




def inference(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	if cfg.model_folder == 'none':
		model, device, n_gpu, vocab_size = instantiate_model(cfg)
	else:
		model = load_model(cfg, cfg.model_folder, device)
	ranking_results_file = cfg.ranking_results.split('/')[-1]
	cfg.model_folder += f'{ranking_results_file}/reprs'
	print(cfg.model_folder)
	if cfg.add:
		res_folder_base += f'_{cfg.add}'

	os.makedirs(cfg.model_folder, exist_ok=True)

	print('Loading data...')

	# calculate maximum lengths
	if cfg.dataset == "robust04":
		max_query_len = cfg.robust04.max_length
		max_complete_len = -1
		max_doc_len = cfg.robust04.max_length
	elif cfg.dataset == "msmarco":
		max_query_len = cfg.msmarco.max_query_len
		max_complete_len = cfg.msmarco.max_complete_len
		max_doc_len = None
	else:
		raise ValueError("\'dataset\' not properly set!: Expected \'robust04\' or \'msmarco\', but got \'" + cfg.dataset  + "\' instead")
	query_fi = File(cfg.queries)
	docs_fi = FileInterface(cfg.docs)
	
	#dataloader = RankingResultsTest(cfg.ranking_results, query_fi , docs_fi, cfg.batch_size_test, max_query_len=max_query_len, max_complete_len=max_complete_len, max_doc_len=max_doc_len, rerank_top_N = cfg.rerank_top_N, device=device)

	dataset = SingleSequential(cfg.ranking_results, query_fi , docs_fi, max_query_len=max_query_len, max_complete_len=max_complete_len, max_doc_len=max_doc_len)
	dataloader = DataLoader(dataset, batch_size=cfg.batch_size_test, collate_fn=collate_fn_bert_interaction_single, num_workers=1)
	print('getting representations...')
	with torch.no_grad():
		model.eval()
		reprs_bert_interaction(model, dataloader, device, f'{cfg.model_folder}/attentions')


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()

	if "model_folder" not in cl_cfg or "docs" not in cl_cfg or "queries" not in cl_cfg or "ranking_results" not in cl_cfg:
		raise ValueError("usage: inference.py model_folder=MODEL_FOLDER docs=DOCS_PATH queries=DOCS_PATH ranking_results=RANKING_RESULTS_PATH")
	if cl_cfg.model_folder != 'none':
		# getting model config
		cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
		# merging both
		cfg = OmegaConf.merge(cfg_load, cl_cfg)
		cfg_default = OmegaConf.load('config.yaml')
		cfg = OmegaConf.merge(cfg_default, cfg)
	else:
		cfg_default = OmegaConf.load('config.yaml')
		cfg = OmegaConf.merge(cfg_default, cl_cfg)

	if cfg.rerank_top_N is None:
		cfg.rerank_top_N = -1

	if cfg.report_top_N is None:
		cfg.report_top_N = -1
	print(cfg.pretty())
	inference(cfg)
