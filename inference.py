
import matplotlib
import torch
from omegaconf import OmegaConf

matplotlib.use('Agg')
import os
import pickle

from enlp.utils import write_ranking, write_ranking_trec, plot_histogram_of_latent_terms, plot_ordered_posting_lists_lengths, load_model, instantiate_model
from enlp.metrics import MRR, MAPTrec
from enlp.file_interface import File
from enlp.run_model import test
from enlp.dataset import RankingResultsTest

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
		model = load_model(cfg, cfg.model_folder, device, old_transformer_lib_model=False)

	print(model)
	if cfg.fold is not None:
		fold = pickle.load(open(f'{cfg.robust_ranking_results_strong}.p', 'rb'))
		print('train', sorted(fold[int(cfg.fold)][0]))
		fold = fold[int(cfg.fold)][1]
		#fold = [303, 307, 310, 317, 318, 327, 330, 331, 336, 341, 347, 349, 350, 366, 368, 369, 370, 372, 374, 378, 387, 390, 395, 396, 397, 400, 401, 403, 408, 419, 431, 433, 434, 439, 610, 611, 612, 614, 622, 630, 632, 635, 639, 640, 645, 659, 686, 697]
		#fold = [301, 302, 304, 305, 306, 308, 309, 311, 312, 313, 314, 315, 316, 319, 320, 321, 322, 323, 324, 325, 326, 328, 329, 332, 333, 334, 335, 337, 338, 339, 340, 342, 343, 344, 345, 346, 348, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 367, 371, 373, 375, 376, 377, 379, 380, 381, 382, 383, 384, 385, 386, 388, 389, 391, 392, 393, 394, 398, 399, 402, 404, 405, 406, 407, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 432, 435, 436, 437, 438, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 601, 602, 603, 604, 605, 606, 607, 608, 609, 613, 615, 616, 617, 618, 619, 620, 621, 623, 624, 625, 626, 627, 628, 629, 631, 633, 634, 636, 637, 638, 641, 642, 643, 644, 646, 647, 648, 649, 650, 651, 652, 653, 654, 655, 656, 657, 658, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 671, 673, 674, 675, 676, 677, 678, 679, 680, 681, 682, 683, 684, 685, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696, 698, 699, 700]
		#fold = [str(f) for f in fold]

		print(f'Evaluating fold {cfg.fold}:', sorted(fold))
	else:
		fold = None
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
		cfg.model_folder += f'/{res_folder_base}/'
		print(cfg.model_folder)
		metric = MAPTrec(cfg.trec_eval, cfg.qrels, cfg.max_rank, add_params=add_params)
	elif cfg.metric == 'none':

		res_folder_base = cfg.queries.split('/')[-1]
		res_folder_base += "_rerank_top_" + str(cfg.rerank_top_N) if cfg.rerank_top_N != -1 else ""
		cfg.model_folder += f'/{res_folder_base}/'

		metric = None
	else:
		NotImplementedError(f'Dataset {cfg.dataset} not implemented!')
	if cfg.add:
		res_folder_base += f'_{cfg.add}'

	os.makedirs(cfg.model_folder, exist_ok=True)

	print('Loading data...')

	# calculate maximum lengths
	if cfg.dataset == "robust04":
		max_query_len = cfg.robust04.max_len
		max_complete_len = -1
		max_doc_len = cfg.robust04.max_len
	elif cfg.dataset == "msmarco":
		max_query_len = cfg.msmarco.max_query_len
		max_complete_len = cfg.msmarco.max_complete_len
		max_doc_len = None
	else:
		raise ValueError("\'dataset\' not properly set!: Expected \'robust04\' or \'msmarco\', but got \'" + cfg.dataset  + "\' instead")
	query_fi = File(cfg.queries)
	docs_fi = File(cfg.docs)
	dataloaders = {}
	print("Reranking top:", cfg.rerank_top_N)
	dataloaders['test'] = RankingResultsTest(cfg.ranking_results, query_fi , docs_fi, cfg.batch_size_test, max_query_len=max_query_len,
		max_complete_len=max_complete_len, max_doc_len=max_doc_len, rerank_top_N = cfg.rerank_top_N, indices=fold, device=device)

	print('testing...')

	with torch.no_grad():
		model.eval()
		print(f'{cfg.model_folder}/ranking')
		metric_score, scores, q_ids = test(model, 'test', dataloaders, device, 0, metric=metric, writer=None, model_folder=cfg.model_folder)
		metric.score(scores, q_ids, save_path=f'{cfg.model_folder}/ranking')
		if metric_score:
			print(f'{res_folder_base} {metric.name}:\t{metric_score}\n')
			metrics_file_path = f'{cfg.model_folder}/ranking_results.txt'
			with open(metrics_file_path, 'w') as out:
				out.write(f'{res_folder_base} {metric.name}:\t{metric_score}\n')


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()

	if "model_folder" not in cl_cfg or "docs" not in cl_cfg or "queries" not in cl_cfg or "ranking_results" not in cl_cfg or "metric" not in cl_cfg:
		raise ValueError("usage: inference.py model_folder=MODEL_FOLDER docs=DOCS_PATH queries=DOCS_PATH ranking_results=RANKING_RESULTS_PATH metric=METRIC [OPTIONAL]qrels=QRELS_PATH")
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

	print(cfg.pretty())
	inference(cfg)
