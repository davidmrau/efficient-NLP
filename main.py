import torch
from snrm import SNRM
from torch import nn
from run_model import train
from torch.optim import Adam
from bert_based import BERT_based
import os
from datetime import datetime
from omegaconf import OmegaConf
from dataset import get_data_loaders
import shutil
from utils import load_glove_embeddings, get_model_folder_name, get_pretrained_BERT_embeddings


# from transformers import BertConfig, BertForPreTraining, BertTokenizer
from utils import str2lst
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

def exp(cfg):
	# printing params
	print(cfg.pretty())

	# select device depending on availability and user's setting
	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# define which embeddings to load, depending on params
	if cfg.embedding == 'glove':
		embedding_parameters =  load_glove_embeddings(cfg.glove_embedding_path)

	elif cfg.embedding == 'bert':
		embedding_parameters = get_pretrained_BERT_embeddings()
	else:
		if cfg.embedding != "random":
			raise RuntimeError('Define pretrained embeddings ! {bert/glove}')
		cfg.embedding = 'bert'
		embedding_parameters = None
	print('Initializing model...')

	# initialize model according to params (SNRM or BERT-like Transformer Encoder)
	if cfg.model == "snrm":
		model = SNRM(hidden_sizes=str2lst(str(cfg.snrm.hidden_sizes)),
		sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_parameters=embedding_parameters,
		embedding_dim = cfg.snrm.embedding_dim, vocab_size = cfg.vocab_size, dropout_p=cfg.snrm.dropout_p,
		n_gram_model = cfg.snrm.n_gram_model, large_out_biases = cfg.large_out_biases)

	elif cfg.model == "tf":
		model = BERT_based( hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
		sparse_dimensions = cfg.sparse_dimensions, num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = 150,
		vocab_size = cfg.vocab_size, embedding_parameters = embedding_parameters, pooling_method = cfg.tf.pooling_method,
		large_out_biases = cfg.large_out_biases)

	# move model to device
	model = model.to(device=device)
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)


	print(model)
	# initialize tensorboard
	writer = SummaryWriter(log_dir=f'{cfg.model_folder}/tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')

	print('Loading data...')
	# initialize dataloaders
	#

	dataloaders = get_data_loaders(cfg.triplets_file_train, cfg.docs_file_train,
	cfg.query_file_train, cfg.query_file_val, cfg.docs_file_val, cfg.batch_size, debug=cfg.debug)
	print('done')
	# initialize loss function
	loss_fn = nn.MarginRankingLoss(margin = 1).to(device)

	# initialize optimizer
	optim = Adam(model.parameters(), lr=cfg.lr)
	print('Start training...')
	# train the model
	model = train(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device,
	cfg.model_folder, cfg.qrels_val, cfg.dataset_path, cfg.sparse_dimensions, top_results=cfg.top_results,
	l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar, patience = cfg.patience, MaxMRRRank=cfg.MaxMRRRank, eval_every = cfg.eval_every)

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

	completed_model_folder = os.path.join(cfg.experiments_dir, model_folder)

	temp_model_folder = os.path.join(cfg.experiments_dir, cfg.temp_exp_prefix + model_folder)
	print(completed_model_folder)
	if os.path.isdir(completed_model_folder):
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
	os.renames(temp_model_folder, completed_model_folder)
