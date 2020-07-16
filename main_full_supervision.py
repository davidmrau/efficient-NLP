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


from enlp.file_interface import FileInterface
from enlp.utils import get_model_folder_name, _getThreads, instantiate_model, gen_folds
from enlp.metrics import MAPTrec
from enlp.utils import offset_dict_len
from enlp.run_model import run
from enlp.dataset import get_data_loaders_robust_strong
import pickle as p
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

	metric = MAPTrec(cfg.trec_eval, cfg.robust_qrel_test, cfg.max_rank)
	print('done')


	dataset_len = offset_dict_len(cfg.robust_ranking_results_strong)

	folds = gen_folds(dataset_len, cfg.num_folds)
	docs_fi = FileInterface(cfg.robust_docs)
	query_fi = FileInterface(cfg.robust_query_test)
	ranking_results_fi = FileInterface(cfg.robust_ranking_results_strong)

	print('Start training...')
	metric_scores = list()
	fold_count = -1

	for i, (indices_train, indices_test) in enumerate(folds):

		fold_count += 1
		completed_model_folder = f'{completed_model_folder_general}/{i}/'
		temp_model_folder = f'{temp_model_folder_general}/{i}/'
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

		print("Training :", temp_model_folder)
		os.makedirs(temp_model_folder, exist_ok=True)

		# save config
		OmegaConf.save(cfg, f'{temp_model_folder}/config.yaml')
		# set model_folder
		cfg.model_folder = temp_model_folder


		print(f'Running fold {fold_count}')

		# initialize model according to params (SNRM or BERT-like Transformer Encoder)

		writer = SummaryWriter(log_dir=f'{cfg.model_folder}/tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')
		model, device, n_gpu = instantiate_model(cfg)
		# initialize loss function
		loss_fn = nn.MarginRankingLoss(margin = 1).to(device)


		# initialize optimizer
		optim = Adam(model.parameters(), lr=cfg.lr)

		dataloaders = get_data_loaders_robust_strong(cfg, indices_train, indices_test, docs_fi, query_fi, ranking_results_fi, cfg.sample_random)
		data = dataloaders['test']
		data.reset()
		
		metric_score, total_trained_samples = run(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
								   cfg.model_folder, l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience = cfg.patience,
								   samples_per_epoch_train = cfg.samples_per_epoch_train, samples_per_epoch_val=cfg.samples_per_epoch_val, bottleneck_run = cfg.bottleneck_run,
								   log_every_ratio = cfg.log_every_ratio, max_rank = cfg.max_rank, metric = metric, sparse_dimensions = cfg.sparse_dimensions, validate=False,
								   max_samples_per_gpu = cfg.max_samples_per_gpu, n_gpu = n_gpu, telegram=cfg.telegram)

		metric_scores.append(metric_score)

		os.renames(temp_model_folder, completed_model_folder)
	open(f'{completed_model_folder_general}/final_score.txt', 'w').write(f'Av {metric.name} Supervised', np.mean(metric_scores), 0).close()

	p.dump(folds, open(f'{compled_model_folder_general}/folds.p', 'wb'))
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
