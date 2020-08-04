
import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use('Agg')
import os


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


	

	# load data

		#metric = MAPTrec(cfg.trec_eval, cfg.qrels, cfg.max_rank, add_params='-l 2')
	#metric = MRR(cfg.msmarco_qrels_test, cfg.max_rank)
	

	
	if cfg.metric == 'mrr':
		metric = MRR(cfg.qrels, 10)
	elif cfg.metric == 'map':
		
		res_folder_base = cfg.qrels.split('/')[-1]
		cfg.model_folder += f'/{res_folder_base}/'
		metric = MAPTrec(cfg.trec_eval, cfg.qrels, cfg.max_rank, save_all_path=cfg.model_folder)
	elif cfg.metric == 'none':

		res_folder_base = cfg.queries.split('/')[-1]
		cfg.model_folder += f'/{res_folder_base}/'
		metric = None
	else:
		NotImplementedError(f'Dataset {cfg.dataset} not implemented!')

	
	os.makedirs(cfg.model_folder, exist_ok=True)

	print('Loading data...')


	dataloaders = {}
	dataloaders['test'] = RankingResultsTest(cfg.ranking_results, cfg.queries, cfg.docs, cfg.batch_size_test, max_query_len=cfg.msmarco.max_query_len, max_complete_length=cfg.msmarco.max_complete_length)

	
	print('testing...')
	if metric:
		metric_score = test(model, 'test', dataloaders, device, cfg.max_rank, 0, metric=metric, writer=None, model_folder=cfg.model_folder)
		print(f'{res_folder_base} {metric.name}:\t{metric_score}\n')

		metrics_file_path = f'{cfg.model_folder}/ranking_results.txt'
		with open(metrics_file_path, 'w') as out:
			out.write(f'{res_folder_base} {metric.name}:\t{metric_score}\n')
	else:
		scores, q_ids = test(model, 'test', dataloaders, device, cfg.max_rank, 0, metric=None, writer=None, model_folder=cfg.model_folder)
		write_ranking_trec(scores, q_ids, cfg.model_folder + '/ranking.trec')


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.docs or not cl_cfg.queries or not cl_cfg.qrels or not cl_cfg.ranking_results or not cl_cfg.metric :
		raise ValueError("usage: inference.py model_folder=MODEL_FOLDER qrels=QRELS_PATH docs=DOCS_PATH queries=DOCS_PATH ranking_results=RANKING_RESULTS_PATH metric=METRIC")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	cfg_default = OmegaConf.load('config.yaml')
	cfg = OmegaConf.merge(cfg_default, cfg)
	inference(cfg)
