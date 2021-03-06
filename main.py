
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from enlp.dataset import get_data_loaders_robust, get_data_loaders_msmarco
from enlp.metrics import MAPTrec
from enlp.run_model import run
from enlp.utils import get_model_folder_name, _getThreads, instantiate_model, get_max_samples_per_gpu, load_model


def exp(cfg):
	# set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	if cfg.bottleneck_run:
		print("!! RUNNING bottleneck CHECK !!")
		cfg.num_epochs = 1


	# initialize model according to params (SNRM or BERT-like Transformer Encoder)

	model, device, n_gpu, vocab_size = instantiate_model(cfg)

	# load model
	if cfg.load:
		model = load_model(cfg, cfg.load, device)

	# set seeds
	torch.manual_seed(cfg.seed)
	np.random.seed(cfg.seed)
	random.seed(cfg.seed)

	# printing params
	print(cfg.pretty())

	avail_threads = _getThreads()

	# if not specified, all available threads will be used
	if cfg.num_workers == -1:
		# print("Using all avaialbe Theads")
		cfg.num_workers = avail_threads
	# if more than available threads are requested, then only using all available threads
	elif cfg.num_workers > avail_threads:
		cfg.num_workers = avail_threads

	print(f"Using {cfg.num_workers} exra thread(s) for each dataset.")

	print(model)
	# initialize tensorboard

	writer = SummaryWriter(log_dir=f'{cfg.model_folder}/tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')


	if isinstance(model, torch.nn.DataParallel):
		model_type = model.module.model_type
	else:
		model_type = model.model_type

	cfg.model_type = model_type


	print('Loading data...')
	# initialize dataloaders
	if cfg.dataset == 'msmarco':
		dataloaders = get_data_loaders_msmarco(cfg, device=device)
		metric = MAPTrec(cfg.trec_eval, cfg.msmarco_qrel_test, cfg.max_rank, add_params='-l 2')
		max_len = cfg.msmarco.max_complete_len
		#metric = MRR(cfg.msmarco_qrels_test, cfg.max_rank)
	elif cfg.dataset == 'robust04':
		dataloaders = get_data_loaders_robust(cfg, device=device)
		metric = MAPTrec(cfg.trec_eval, cfg.robust_qrel_test, cfg.max_rank)
		max_len = cfg.robust04.max_length
	else:
		NotImplementedError(f'Dataset {cfg.dataset} not implemented!')
	print('done')

	if model_type == "bert-interaction" or model_type == "rank_prob" or model_type == 'score-interaction':
		loss_fn = nn.CrossEntropyLoss()
	else:
		# initialize loss function
		loss_fn = nn.MarginRankingLoss(margin = cfg.margin).to(device)


	# initialize optimizer
	optim = Adam(model.parameters(), lr=cfg.lr)


	# if max_samples_per_gpu is not set (-1), then dynamically calculate it
	# if cfg.max_samples_per_gpu == -1:
	# 	cfg.max_samples_per_gpu = get_max_samples_per_gpu(model, device, n_gpu, optim, loss_fn, max_len, vocab_size)
	# 	print("max_samples_per_gpu, was not defined. Dynamically calculated to be equal to :", cfg.max_samples_per_gpu)

	# if model_type == "bert-interaction":
	# 	cfg.batch_size_test = n_gpu * cfg.max_samples_per_gpu
	# else:
	# 	cfg.batch_size_test = n_gpu * 3 * cfg.max_samples_per_gpu

	print('Printing Parameters...')
	# printing params
	print(cfg.pretty())
	# save config
	OmegaConf.save(cfg, f'{cfg.model_folder}/config.yaml')

	print('Starting training...')
	# train the model
	_, _ = run(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
	cfg.model_folder, l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience=cfg.patience,
	samples_per_epoch_train=cfg.samples_per_epoch_train, samples_per_epoch_val=cfg.samples_per_epoch_val, bottleneck_run=cfg.bottleneck_run,
	log_every_ratio=cfg.log_every_ratio, metric=metric, telegram=cfg.telegram, sub_batch_size=cfg.sub_batch_size)


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.load:
		cfg_load = OmegaConf.load(f'config.yaml')
		# create model string, depending on the model
	else:
		cfg_load = OmegaConf.load(f'{cl_cfg.load}/config.yaml')




	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)

	if cl_cfg.load:
		model_folder = cl_cfg.load
		model_folder += f'_FT_{cl_cfg.dataset}'
	elif not cl_cfg.model_folder:
		model_folder = get_model_folder_name(cfg)
	else:
		model_folder = cl_cfg.model_folder



	if not cfg.dataset:
		raise ValueError('No Dataset chosen!')

	if cl_cfg.add:
		model_folder += f'_{cl_cfg.add}'

	if cfg.debug:
		model_folder += "_DEBUG"

	if cfg.bottleneck_run:
		model_folder = "BOTTLENECK_RUN_" + model_folder

	completed_model_folder = os.path.join(cfg.experiments_dir, model_folder)

	temp_model_folder = os.path.join(cfg.experiments_dir, cfg.temp_exp_prefix + model_folder)

	print("Complete Model Path: ", completed_model_folder)
	if os.path.isdir(completed_model_folder):

		if cfg.bottleneck_run:

			while(os.path.isdir(completed_model_folder)):
				model_folder +=  "."
				completed_model_folder = os.path.join(cfg.experiments_dir, model_folder)
				temp_model_folder = os.path.join(cfg.experiments_dir, cfg.temp_exp_prefix + model_folder)
		else:
			if cfg.debug:
				shutil.rmtree(completed_model_folder)
				print("Deleted model folder and starting from scratch.")
			else:
				print("\nExperiment Directory:\n",completed_model_folder,"\nis already there, skipping the experiment !!!")
				exit()

	elif os.path.isdir(temp_model_folder):
		print("Incomplete experiment directory found :\n", temp_model_folder)
		shutil.rmtree(temp_model_folder)
		print("Deleted it and starting from scratch.")

	print("Training :", model_folder)
	os.makedirs(temp_model_folder, exist_ok=True)
	os.makedirs(temp_model_folder + '/rankings', exist_ok=True)

	# set model_folder
	cfg.model_folder = temp_model_folder

	exp(cfg)

	# after the training is done, we remove the temp prefix from the dir name
	print("Training completed! Changing from temporary name to final name.")
	print("--------------------------------------------------------------------------------------")
	os.renames(temp_model_folder, completed_model_folder)
