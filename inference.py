
import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use('Agg')
import os
import pickle

from enlp.utils import write_ranking, write_ranking_trec, plot_histogram_of_latent_terms, \
	plot_ordered_posting_lists_lengths, load_model
from enlp.metrics import MRR, MAPTrec
from enlp.file_interface import FileInterface
from enlp.run_model import test
from enlp.dataset import RankingResultsTest

""" Run online inference (for test set) without inverted index
"""




def inference(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')


	model = load_model(cfg, cfg.model_folder, device)




	# set minimum length in case of snrm
	min_len = 0
	if cfg.model == "snrm":
		min_len = cfg.snrm.n

	
	if cfg.metric == 'mrr':
		metric = MRR(cfg.qrels, 10)
	elif cfg.metric == 'map':
		add_params = '-l 2' if cfg.dataset == 'msmarco' else ''	
		res_folder_base = cfg.ranking_results.split('/')[-1]
		res_folder_base += "_rerank_top_" + str(cfg.rerank_top_N) if cfg.rerank_top_N != -1 else ""
		res_folder_base += "_report_top_" + str(cfg.report_top_N) if cfg.report_top_N != -1 else ""
		cfg.model_folder += f'/{res_folder_base}/'
		metric = MAPTrec(cfg.trec_eval, cfg.qrels, cfg.max_rank, save_all_path=cfg.model_folder, add_params=add_params)
	elif cfg.metric == 'none':

		res_folder_base = cfg.queries.split('/')[-1]
		res_folder_base += "_rerank_top_" + str(cfg.rerank_top_N) if cfg.rerank_top_N != -1 else ""
		res_folder_base += "_report_top_" + str(cfg.report_top_N) if cfg.report_top_N != -1 else ""
		cfg.model_folder += f'/{res_folder_base}/'

		metric = None
	else:
		NotImplementedError(f'Dataset {cfg.dataset} not implemented!')
	
	os.makedirs(cfg.model_folder, exist_ok=True)

	print('Loading data...')

	# calculate maximum lengths
	if cfg.dataset == "robust04":
		max_query_len = cfg.robust04.max_length
		max_complete_length = -1 
		max_doc_len = cfg.robust04.max_length
	elif cfg.dataset == "msmarco":
		max_query_len = cfg.msmarco.max_query_len
		max_complete_length = cfg.msmarco.max_complete_length
		max_doc_len = None
	else:
		raise ValueError("\'dataset\' not properly set!: Expected \'robust04\' or \'msmarco\', but got \'" + cfg.dataset  + "\' instead")

	dataloaders = {}
	dataloaders['test'] = RankingResultsTest(cfg.ranking_results, cfg.queries, cfg.docs, cfg.batch_size_test, max_query_len=max_query_len,
		max_complete_length=max_complete_length, max_doc_len=max_doc_len, rerank_top_N = cfg.rerank_top_N)

	print('testing...')

	with torch.no_grad():
		model.eval()
		if metric:
			metric_score = test(model, 'test', dataloaders, device, cfg.max_rank, 0, metric=metric, writer=None, model_folder=cfg.model_folder, report_top_N=cfg.report_top_N)
			print(f'{res_folder_base} {metric.name}:\t{metric_score}\n')

			metrics_file_path = f'{cfg.model_folder}/ranking_results.txt'
			with open(metrics_file_path, 'w') as out:
				out.write(f'{res_folder_base} {metric.name}:\t{metric_score}\n')
		else:
			scores, q_ids = test(model, 'test', dataloaders, device, cfg.max_rank, 0, metric=None, writer=None, model_folder=cfg.model_folder, report_top_N=cfg.report_top_N)
			write_ranking_trec(scores, q_ids, cfg.model_folder + '/ranking.trec')
			write_ranking(scores, q_ids, cfg.model_folder + '/ranking.tsv')


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()

	if "model_folder" not in cl_cfg or "docs" not in cl_cfg or "queries" not in cl_cfg or "ranking_results" not in cl_cfg or "metric" not in cl_cfg:
		raise ValueError("usage: inference.py model_folder=MODEL_FOLDER docs=DOCS_PATH queries=DOCS_PATH ranking_results=RANKING_RESULTS_PATH metric=METRIC [OPTIONAL]qrels=QRELS_PATH")

	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	cfg_default = OmegaConf.load('config.yaml')
	cfg = OmegaConf.merge(cfg_default, cfg)

	if cfg.rerank_top_N is None:
		cfg.rerank_top_N = -1 

	if cfg.report_top_N is None:
		cfg.report_top_N = -1 

	inference(cfg)
