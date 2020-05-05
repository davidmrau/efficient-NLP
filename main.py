
from torch import nn
from run_model import train, evaluate
from torch.optim import Adam
import os
from datetime import datetime
from omegaconf import OmegaConf
from dataset import get_data_loaders_robust, get_data_loaders_msmarco
import shutil
from utils import get_model_folder_name, _getThreads, instantiate_model
from metrics import MRR, MAPTrec
from utils import plot_histogram_of_latent_terms, plot_ordered_posting_lists_lengths
from torch.utils.tensorboard import SummaryWriter


def exp(cfg):
	# printing params
	print(cfg.pretty())

	if cfg.bottleneck_run:
		print("!! RUNNING bottleneck CHECK !!")
		cfg.samples_per_epoch = 100
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
		dataloaders = get_data_loaders_msmarco(cfg)
		metric = MRR(cfg.msmarco_qrels_val, cfg.max_rank)
	elif cfg.dataset == 'robust04':
		dataloaders = get_data_loaders_robust(cfg)
		metric = MAPTrec(cfg.trec_eval, cfg.robust_qrel_test, cfg.max_rank)
	else:
		NotImplementedError(f'Dataset {cfg.dataset} not implemented!')
	print('done')
	# initialize loss function
	loss_fn = nn.MarginRankingLoss(margin = 1).to(device)


	# initialize optimizer
	optim = Adam(model.parameters(), lr=cfg.lr)
	print('Start training...')
	# train the model
	model, metric_score, total_trained_samples = train(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
	cfg.model_folder, l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience = cfg.patience,
	samples_per_epoch_train = cfg.samples_per_epoch_train, samples_per_epoch_val=cfg.samples_per_epoch_val, debug = cfg.debug, bottleneck_run = cfg.bottleneck_run,
	log_every_ratio = cfg.log_every_ratio, max_rank = cfg.max_rank, metric = metric, sparse_dimensions = cfg.sparse_dimensions)

	_, q_repr, d_repr, q_ids, _, metric_score = evaluate(model, 'test', dataloaders, device, cfg.max_rank, writer, total_trained_samples, metric)




	# plot stats
	plot_ordered_posting_lists_lengths(cfg.model_folder, q_repr, 'query')
	plot_histogram_of_latent_terms(cfg.model_folder, q_repr, cfg.sparse_dimensions, 'query')
	plot_ordered_posting_lists_lengths(cfg.model_folder, d_repr, 'docs')
	plot_histogram_of_latent_terms(cfg.model_folder, d_repr, cfg.sparse_dimensions, 'docs')


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
