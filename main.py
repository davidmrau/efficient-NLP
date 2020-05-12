
from torch import nn
from run_model import run
from torch.optim import Adam
import os
import torch
from datetime import datetime
from omegaconf import OmegaConf
from dataset import get_data_loaders_robust, get_data_loaders_msmarco
import shutil
import numpy as np
from utils import get_model_folder_name, _getThreads, instantiate_model
from metrics import MRR, MAPTrec
from utils import plot_histogram_of_latent_terms, plot_ordered_posting_lists_lengths
from torch.utils.tensorboard import SummaryWriter
import random

def exp(cfg):
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


	# initialize model according to params (SNRM or BERT-like Transformer Encoder)

	model, device = instantiate_model(cfg)


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

	print(f"Using {cfg.num_workers} exra thread(s) for each dataset.")

	print(model)
	# initialize tensorboard

	writer = SummaryWriter(log_dir=f'{cfg.model_folder}/tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')

	print('Loading data...')
	# initialize dataloaders
	if cfg.dataset == 'msmarco':
		dataloaders = get_data_loaders_msmarco(cfg)
		metric = MRR(cfg.msmarco_qrels_val, cfg.max_rank)
	elif cfg.dataset == 'robust04':
		dataloaders = get_data_loaders_robust(cfg)
		metric = MAPTrec(cfg.trec_eval, cfg.robust_qrel_test, cfg.max_rank)
	else:
		NotImplementedError(f'Dataset {cfg.dataset} not implemented!')
	print('done')
	# initialize loss function
	loss_fn = nn.MarginRankingLoss(margin = cfg.margin).to(device)


	# initialize optimizer
	optim = Adam(model.parameters(), lr=cfg.lr)
	print('Start training...')
	# train the model
	model, metric_score, total_trained_samples = run(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
	cfg.model_folder, l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience = cfg.patience,
	samples_per_epoch_train = cfg.samples_per_epoch_train, samples_per_epoch_val=cfg.samples_per_epoch_val, bottleneck_run = cfg.bottleneck_run,
	log_every_ratio = cfg.log_every_ratio, max_rank = cfg.max_rank, metric = metric, sparse_dimensions = cfg.sparse_dimensions, always_correct=cfg.always_correct)






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
			print("\nExperiment Directory:\n",completed_model_folder,"\nis already there, skipping the experiment !!!")
			exit()

	elif os.path.isdir(temp_model_folder):
		print("Incomplete experiment directory found :\n", temp_model_folder)
		shutil.rmtree(temp_model_folder)
		print("Deleted it and starting from scratch.")

	print("Training :", model_folder)
	os.makedirs(temp_model_folder, exist_ok=True)

	# save config
	OmegaConf.save(cfg, f'{temp_model_folder}/config.yaml')
	# set model_folder
	cfg.model_folder = temp_model_folder

	exp(cfg)

	# after the training is done, we remove the temp prefix from the dir name
	print("Training completed! Changing from temporary name to final name.")
	print("--------------------------------------------------------------------------------------")
	os.renames(temp_model_folder, completed_model_folder)
