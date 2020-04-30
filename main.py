import torch

from torch import nn
from run_model import train
from torch.optim import Adam
import os
from datetime import datetime
from omegaconf import OmegaConf
from dataset import get_data_loaders_robust, get_data_loaders_msmarco
import shutil
from utils import get_model_folder_name, _getThreads, instantiate_model
from metrics import MRR, MAPTrec

# from transformers import BertConfig, BertForPreTraining, BertTokenizer
from utils import str2lst
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer


def exp(cfg):
	# printing params
	print(cfg.pretty())


	if cfg.bottleneck_run:
		print("!! RUNNING bottleneck CHECK !!")
		cfg.eval_every = 100
		cfg.num_epochs = 1


	# initialize model according to params (SNRM or BERT-like Transformer Encoder)

	model, device = instantiate_model(cfg)

	avail_threads = _getThreads()

	# if not specified, all available threads will be used
	if cfg.num_workers == -1:
		# print("Using all avaialbe Theads")
		cfg.num_workers = avail_threads
	# if more than available threads are requested, then only using all available threads
	elif cfg.num_workers > avail_threads:
		cfg.num_workers = avail_threads

	print(f"Using {cfg.num_workers} Threads.")

	print(model)
	# initialize tensorboard

	writer = SummaryWriter(log_dir=f'{cfg.model_folder}/tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')

	print('Loading data...')
	# initialize dataloaders
	if cfg.dataset == 'msmarco':
		dataloaders = get_data_loaders_msmarco(cfg.triplets_file_train, cfg.docs_file_train,
		cfg.query_file_train, cfg.query_file_val, cfg.docs_file_val, cfg.batch_size, cfg.num_workers, debug=cfg.debug)
	elif cfg.dataset == 'robust':
		dataloaders = get_data_loaders_robust(cfg.robust_ranking_results_file, cfg.robust_docs_file,
		cfg.robust_query_file, cfg.batch_size, cfg.num_workers, cfg.sampler, cfg.target, debug=cfg.debug)
	print('done')
	# initialize loss function
	loss_fn = nn.MarginRankingLoss(margin = 1).to(device)



	if cfg.metric == 'map':
		metric = MAPTrec(cfg.trec_eval, cfg.robust_qrel_test, cfg.max_rank)
	elif cfg.metric == 'mrr':
		metric = MRR(cfg.qrels_val, cfg.max_rank)
	else:
		raise NotImplementedError(f'Metric {cfg.metric} not implemented')


	# initialize optimizer
	optim = Adam(model.parameters(), lr=cfg.lr)
	print('Start training...')
	# train the model
	model = train(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
	cfg.model_folder, cfg.sparse_dimensions, metric, max_rank=cfg.max_rank,
	l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience = cfg.patience, eval_every = cfg.eval_every, debug = cfg.debug, bottleneck_run = cfg.bottleneck_run)

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	cfg_load = OmegaConf.load(f'config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)

	# create model string, depending on the model
	if not cl_cfg.model_folder:
		model_folder = get_model_folder_name(cfg)
	else:
		model_folder = cl_cfg.model_folder


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
