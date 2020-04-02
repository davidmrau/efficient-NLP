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
		# # word2idx = generate_word2idx_dict_from_glove
		# word2idx = read_pickle(cfg.glove_word2idx_path)

	elif cfg.embedding == 'bert':
		embedding_path = 'bert'
		# # load BERT's BertTokenizer
		# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
		# # use BERT's word to ID
		# word2idx = tokenizer.vocab
		# del tokenizer
	else:
		raise RuntimeError('Define pretrained embeddings ! {bert/glove}')
	print('Initializing model...')

	# initialize model according to params (SNRM or BERT-like Transformer Encoder)
	if cfg.model == "snrm":
		model = SNRM(hidden_sizes=str2lst(str(cfg.snrm.hidden_sizes)),
		sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_path=embedding_path,
		dropout_p=cfg.snrm.dropout_p)

	elif cfg.model == "tf":
		model = BERT_based( hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
		sparse_dimensions = cfg.sparse_dimensions, num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = 150,
		vocab_size = cfg.vocab_size, embedding_path = embedding_path, pooling_method = cfg.tf.pooling_method)

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
	l1_scalar=cfg.l1_scalar, balance_scalar=cfg.balance_scalar)

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	cfg_load = OmegaConf.load(f'config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)

	# create model string, depending on the model
	if not cl_cfg.model_folder:
		if cfg.model == "tf":
			# updating hidden dimensions according to selected embeddings
			if cfg.embedding == "bert":
				cfg.tf.hidden_size=768
			elif cfg.embedding == "glove":
				cfg.tf.hidden_size=300

			model_string=f"{cfg.model}_L_{cfg.tf.num_of_layers}_H_{cfg.tf.num_attention_heads}_D_{cfg.tf.hidden_size}_P{cfg.tf.pooling_method}"
		elif cfg.model == "snrm":
			model_string=f"{cfg.model}_{cfg.snrm.hidden_sizes}"

		else:
			raise ValueError("Model not set properly!:", cfg.model)
		# create experiment directory name
		model_folder = f"experiments/l1_{cfg.l1_scalar}_Emb_{cfg.embedding}_Sparse_{cfg.sparse_dimensions}_bsz_{cfg.batch_size}_lr_{cfg.lr}_{model_string}"
	else:
		model_folder = cl_cfg.model_folder

	print("Training :", model_folder)
	
	if os.path.isdir(model_folder):
		exit()
	os.makedirs(model_folder, exist_ok=True)

	# save config
	OmegaConf.save(cfg, f'{model_folder}/config.yaml')
	# set model_folder
	cfg.model_folder = model_folder
	exp(cfg)
