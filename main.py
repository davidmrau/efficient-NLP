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
		embedding_path = cfg.glove_embedding_path
	elif cfg.embedding == 'bert':
		embedding_path = 'bert'

	# load BERT's BertTokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# use BERT's word to ID
	word2idx = tokenizer.vocab
	print('Initializing model...')

	# initialize model according to params (SNRM or BERT-like Transformer Encoder)
	if cfg.model == "snrm":
		model = SNRM(hidden_sizes=str2lst(str(cfg.snrm.hidden_sizes)),
		sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_path=embedding_path,
		word2idx=word2idx, dropout_p=cfg.snrm.dropout_p, debug=cfg.debug, device=device)

	elif cfg.model == "tf":
		model = BERT_based(  hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
		sparse_dimensions = cfg.sparse_dimensions, vocab_size = 30522,
		num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = 150,
		pretrained_embeddings = cfg.tf.pretrained_embeddings, pooling_method = "CLS", device=device)

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
	dataloaders = get_data_loaders(cfg.dataset_path, cfg.batch_size, debug=cfg.debug)
	print('done')
	# initialize loss function
	loss_fn = nn.MarginRankingLoss(margin = 1).to(device)


	# initialize optimizer
	optim = Adam(model.parameters(), lr=cfg.lr)
	print('Start training...')
	# train the model
	model = train(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device, cfg.model_folder, l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar)

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	cfg_load = OmegaConf.load(f'config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)

	if not cl_cfg.model_folder:
		print('No model folder specified, using timestemp instead.')
		model_folder = f'experiments/{datetime.now().strftime("%Y_%m_%d_%H_%M")}/'
	else:
		model_folder = cl_cfg.model_folder

	os.makedirs(model_folder, exist_ok=True)

	# save config
	OmegaConf.save(cfg, f'{model_folder}/config.yaml')
	# set model_folder
	cfg.model_folder = model_folder
	exp(cfg)
