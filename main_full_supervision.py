import os
import random
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings("ignore")
from enlp.file_interface import File
from enlp.utils import get_model_folder_name, _getThreads, instantiate_model, get_max_samples_per_gpu
from enlp.metrics import MAPTrec
from enlp.utils import offset_dict_len
from enlp.run_model import run
from enlp.dataset import get_data_loaders_robust_strong
import pickle
def exp(cfg, temp_model_folder_general, completed_model_folder_general):




	# set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	# printing params
	print(cfg.pretty())

	if cfg.bottleneck_run:
		print("!! RUNNING bottleneck CHECK !!")
		cfg.samples_per_epoch = 100
		cfg.num_epochs = 1





	# set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	avail_threads = _getThreads()

	# if not specified, all available threads will be used
	if cfg.num_workers == -1:
		# print("Using all avaialbe Theads")
		cfg.num_workers = avail_threads
	# if more than available threads are requested, then only using all available threads
	elif cfg.num_workers > avail_threads:
		cfg.num_workers = avail_threads

	print(f"Using {cfg.num_workers} Threads.")

	# initialize tensorboard
	print('Loading data...')
	# initialize dataloaders
	if cfg.dataset == 'msmarco':
		add_params= '-l 2'
	else:
		add_params = ''

	print('done')
	max_len = 1500


	metric = MAPTrec(cfg.trec_eval, cfg.robust_qrel_test, cfg.max_rank, add_params=add_params)

#	dataset_len = offset_dict_len(cfg.robust_ranking_results_strong)
#	folds = gen_folds(dataset_len, cfg.num_folds)
	if cfg.debug:
		cfg.robust_ranking_results_strong += '_debug'
		folds = [[None, [226]], [None, [16, 222]], [None, [16,152, 222]], [None, None], [None, None]]
	else:
		folds = pickle.load(open(cfg.folds_file, 'rb'))
		#folds = [[None, list(range(44))]]


	docs_fi = File(cfg.robust_docs)
	query_fi = File(cfg.robust_query_test)
	print('Start training...')
	metric_scores = list()

	for i, (indices_train, indices_test) in enumerate(folds):
		print(indices_train, indices_test)
		ranking_results = f'{cfg.robust_ranking_results_strong}_{i}'
		completed_model_folder = f'{completed_model_folder_general}/{i}/'
		temp_model_folder = f'{temp_model_folder_general}/{i}/'
		print("Complete Model Path: ", completed_model_folder)
		if os.path.isdir(completed_model_folder):

			if cfg.bottleneck_run:

				while(os.path.isdir(completed_model_folder)):
					model_folder =  "."
					completed_model_folder = os.path.join(cfg.experiments_dir, model_folder)
					temp_model_folder = os.path.join(cfg.experiments_dir, cfg.temp_exp_prefix + model_folder)
			else:
				print("\nExperiment Directory:\n",completed_model_folder,"\nis already there, skipping the experiment !!!")
				continue

		elif os.path.isdir(temp_model_folder):
			print("Incomplete experiment directory found :\n", temp_model_folder)
			shutil.rmtree(temp_model_folder)
			print("Deleted it and starting from scratch.")

		print("Training :", temp_model_folder)
		os.makedirs(temp_model_folder, exist_ok=True)
		# create folder to store rankings in
		os.makedirs(temp_model_folder + '/rankings', exist_ok=True)

		# save config
		OmegaConf.save(cfg, f'{temp_model_folder}/config.yaml')
		# set model_folder
		cfg.model_folder = temp_model_folder


		print(f'Running fold {i}')

		# initialize model according to params (SNRM or BERT-like Transformer Encoder)

		writer = SummaryWriter(log_dir=f'{cfg.model_folder}/tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')
		model, device, n_gpu, vocab_size = instantiate_model(cfg)

		if isinstance(model, torch.nn.DataParallel):
			model_type = model.module.model_type
		else:
			model_type = model.model_type

		# initialize loss function
		if model_type == "bert-interaction" or model_type == "rank_prob":
			loss_fn = nn.CrossEntropyLoss()
		else:
		# initialize loss function
			loss_fn = nn.MarginRankingLoss(margin = cfg.margin).to(device)

		# initialize optimizer
		optim = Adam(model.parameters(), lr=cfg.lr)




		cfg.model_type = model_type
		# if max_samples_per_gpu is not set (-1), then dynamically calculate it
		#if cfg.max_samples_per_gpu == -1:
			#cfg.max_samples_per_gpu = get_max_samples_per_gpu(model, device, n_gpu, optim, loss_fn, max_len, vocab_size)
			#print("max_samples_per_gpu, was not defined. Dynamically calculated to be equal to :", cfg.max_samples_per_gpu)





		dataloaders = get_data_loaders_robust_strong(cfg, indices_test, query_fi, docs_fi, ranking_results, device=device)
		data = dataloaders['test']
		data.reset()

		metric_score, total_trained_samples = run(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
								   cfg.model_folder, l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience = cfg.patience,
								   samples_per_epoch_train = cfg.samples_per_epoch_train, samples_per_epoch_val=cfg.samples_per_epoch_val, bottleneck_run = cfg.bottleneck_run,
								   log_every_ratio = cfg.log_every_ratio, metric = metric, validate=False,
									telegram=cfg.telegram)

		metric_scores.append(metric_score)

		os.makedirs(completed_model_folder, exist_ok=True)
		os.renames(temp_model_folder, completed_model_folder)
		break
	open(f'{completed_model_folder_general}/final_score.txt', 'w').write(f'Av {metric.name} Supervised {np.mean(metric_scores)}')

	# after the training is done, we remove the temp prefix from the dir name
	print("Training completed! Changing from temporary name to final name.")
	print("--------------------------------------------------------------------------------------")

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	cfg_load = OmegaConf.load(f'config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	if not cfg.dataset:
		raise ValueError('No Dataset chosen!')
	# create model string, depending on the model
	if not cl_cfg.model_folder:
		model_folder = get_model_folder_name(cfg)
	else:
		model_folder = cl_cfg.model_folder


	if cfg.debug:
		model_folder += "_DEBUG"

	if cfg.bottleneck_run:
		model_folder = "BOTTLENECK_RUN_" + model_folder

	if cl_cfg.load_model:
		model_folder += '_fine_tuned'
	else:
		model_folder += '_full_supervision'
	if cl_cfg.add:
		model_folder += f'_{cl_cfg.add}'
	completed_model_folder = os.path.join(cfg.experiments_dir, model_folder)

	temp_model_folder = os.path.join(cfg.experiments_dir, cfg.temp_exp_prefix + model_folder)
	cfg.model_folder = completed_model_folder
	exp(cfg, temp_model_folder, completed_model_folder)
