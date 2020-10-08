import transformers
import csv
import json
import os
import pickle
import random
import subprocess

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

import sys
import random
from enlp.models.rank_model import RankModel
from enlp.models.bert_based import BERT_based
from enlp.models.snrm import SNRM
from enlp.models.bert import BERT_inter
import collections

matplotlib.use('Agg')
#
# from https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
# Thanks!
#
class EarlyStopping(object):
	def __init__(self, mode='min', min_delta=0, patience=10):
		self.mode = mode
		self.min_delta = min_delta
		self.patience = patience
		self.best = None
		self.num_bad_epochs = 0
		self.is_better = None
		self._init_is_better(mode, min_delta)

		self.stop = False

	def step(self, metrics):
		if self.best is None:
			self.best = metrics
			return False

		if np.isnan(metrics):
			self.stop = True
			return True

		if self.is_better(metrics, self.best):
			self.num_bad_epochs = 0
			self.best = metrics
		else:
			self.num_bad_epochs += 1

		if self.num_bad_epochs >= self.patience:
			self.stop = True
			return True

		return False

	def _init_is_better(self, mode, min_delta):

		if mode not in {'min', 'max'}:
			raise ValueError('mode ' + mode + ' is unknown!')

		if mode == 'min':
			self.is_better = lambda a, best: a < best - (best * min_delta / 100)
		if mode == 'max':
			self.is_better = lambda a, best: a > best + (best * min_delta / 100)



class Average(object):

	def __init__(self):
		self.val = None
		self.count = 0

	def step(self, x):
		self.count += 1
		if self.val is None:
			self.val = x
		else:
			self.val = self.val + (x - self.val) / self.count


def split_sizes(dataset_len, train_val_ratio):
	return [math.floor(dataset_len*train_val_ratio), math.ceil(dataset_len*(1-train_val_ratio))]

def split_by_len(dataset_len, ratio):
	rand_index = list(range(dataset_len))
	random.shuffle(rand_index)
	sizes = split_sizes(dataset_len, ratio )
	indices_train = rand_index[:sizes[0]]
	indices_test = rand_index[sizes[0]:]
	return indices_train, indices_test

def split_dataset(train_val_ratio, dataset):
	# split dataset into train and test
	lengths = split_sizes(len(dataset), train_val_ratio)
	train_dataset, validation_dataset = torch.utils.data.dataset.random_split(dataset, lengths)
	return train_dataset, validation_dataset


def file_len(fname):
	""" Get the number of lines from file
	""" # https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
	p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	result, err = p.communicate()
	if p.returncode != 0:
		raise IOError(err)
	return int(result.strip().split()[0])


def get_offset_dict_path(filename):
	return filename.replace('.gz', '') + '.offset_dict.p'

def offset_dict_len(filename):
	return len(read_pickle(get_offset_dict_path(filename)))


def utilize_pretrained_bert(cfg):
	params_to_copy = {}

	# the model might be "tf" or "bert". We n/eed to handle each case dynamically
	model_name = cfg.__getattr__("model")

	load_bert_layers = cfg.__getattr__(model_name).__getattr__("load_bert_layers")
	load_bert_path = cfg.__getattr__(model_name).__getattr__("load_bert_path")

	if isinstance(load_bert_layers, str) and len(load_bert_layers) != 0:
		load_bert_layers = str2lst(str(load_bert_layers))
	elif isinstance(load_bert_layers, int):
		load_bert_layers = [load_bert_layers]
	else:
		load_bert_layers = []

	if len(load_bert_layers) > 0:
		# load state dictionary of the model that we will copy the paramterers from
		if load_bert_path == 'default':
			model = transformers.BertModel.from_pretrained('bert-base-uncased')
		else:
			model =  transformers.BertModel.from_pretrained(load_bert_path)

		#if isinstance(model, torch.nn.DataParallel):
		#	model = model.module
		#model = model.to('cpu')

		model_state_dict = model.state_dict()

		# update the number of layers, depending on the layers that need to be copied
		cfg.__getattr__(model_name)["num_of_layers"] =  max( max(load_bert_layers), cfg.__getattr__(model_name)["num_of_layers"])

		if max(load_bert_layers) > 0:
			# retrieve the number of heads, according to the loaded model
			cfg.__getattr__(model_name)["num_attention_heads"] =  model.config.num_attention_heads



		for layer in load_bert_layers:

			for key in model_state_dict:

				# setting the actual layer index, cause we consider 0 to be embeddings,
				# so first layer will be represented as 1, but has actual index of 0

				if layer == 0:
					check_string = 'embeddings'

					# in case we are loading embeddings, including positional embeddings,
					# we need to adjust the input length limit, and hidden representation size
					if "position_embeddings" in key:

						cfg.__getattr__(model_name)["input_length_limit"] =  model_state_dict[key].size(0)
						cfg.__getattr__(model_name)["hidden_size"] =  model_state_dict[key].size(1)
				else:
					check_string = "." + str(int(layer) - 1) + "."
					# if we are loading at least one bert layer, then we need to copy the number of attention heads

					cfg.__getattr__(model_name)["num_attention_heads"] =  model.config.num_attention_heads
					# also making sure that we have the correct hidden size
					cfg.__getattr__(model_name)["hidden_size"] =  model.config.hidden_size



				if check_string in key:
					params_to_copy[key] =  torch.nn.Parameter(model_state_dict[key])

	return params_to_copy


def instantiate_model(cfg):

	print('Initializing model...')

	if cfg.embedding == 'glove':
		embedding_parameters = load_glove_embeddings(cfg.glove_embedding_path)

	elif cfg.embedding == 'bert':
		if cfg.bert_rel:
			embedding_parameters = load_glove_embeddings(cfg.bert_relevance_embeddings_path)
		else:
			embedding_parameters = get_pretrained_BERT_embeddings()
	else:
		if cfg.embedding != "random":
			raise RuntimeError('Define pretrained embeddings ! {bert/glove}')
		cfg.embedding = 'bert'
		embedding_parameters = None


	if cfg.model == "snrm":
		model = SNRM(hidden_sizes=str2lst(str(cfg.snrm.hidden_sizes)),
					 sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_parameters=embedding_parameters,
					 embedding_dim = cfg.snrm.embedding_dim, vocab_size = cfg.vocab_size, dropout_p=cfg.snrm.dropout_p,
					 n_gram_model = cfg.snrm.n_gram_model, large_out_biases = cfg.large_out_biases)

		vocab_size = model.vocab_size

	elif cfg.model == "rank":
		model = RankModel(hidden_sizes = str2lst(str(cfg.rank_model.hidden_sizes)), embedding_parameters = embedding_parameters,
			embedding_dim = cfg.rank_model.embedding_dim, vocab_size = cfg.vocab_size, dropout_p = cfg.rank_model.dropout_p,
			weights = cfg.rank_model.weights, trainable_weights = cfg.rank_model.trainable_weights)


		vocab_size = model.vocab_size

	elif cfg.model == "bert":
		params_to_copy = utilize_pretrained_bert(cfg)

		model = BERT_inter(hidden_size = cfg.bert.hidden_size, num_of_layers = cfg.bert.num_of_layers,
							num_attention_heads = cfg.bert.num_attention_heads, input_length_limit = cfg.bert.input_length_limit,
							vocab_size = cfg.vocab_size, embedding_parameters = embedding_parameters, params_to_copy = params_to_copy)

		vocab_size = model.encoder.config.vocab_size

	elif cfg.model == "tf":
		params_to_copy = utilize_pretrained_bert(cfg)

		model = BERT_based( hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
							sparse_dimensions = cfg.sparse_dimensions, num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = cfg.tf.input_length_limit,
							vocab_size = cfg.vocab_size, embedding_parameters = embedding_parameters, pooling_method = cfg.tf.pooling_method,
							large_out_biases = cfg.large_out_biases, layer_norm = cfg.tf.layer_norm, act_func = cfg.tf.act_func,
							params_to_copy = params_to_copy)

		vocab_size = model.encoder.config.vocab_size

	# select device depending on availability and user's setting
	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
		# move model to device
		n_gpu = torch.cuda.device_count()
	else:
		device = torch.device('cpu')
		n_gpu = 1


	if n_gpu > 1:
		print("Using", n_gpu, "GPUs!")
		model = nn.DataParallel(model)

	model = model.to(device=device)

	return model, device, n_gpu, vocab_size







def get_pretrained_BERT_embeddings():
	bert = transformers.BertModel.from_pretrained('bert-base-uncased')
	return bert.embeddings.word_embeddings.weight


def prepro_glove_embeddings(path = "data/embeddings/glove.6B.300d.txt"):
	""" Load Glove embeddings from file
	"""
	embeddings = []
	word2idx = {}
	idx2word = {}

	with open(path) as f:
		index = 0
		line = f.readline()
		while(line):
			line = line.split()
			word = line[0]

			word2idx[word] = index
			idx2word[index] = word

			embeddings.append( line[-300:] )

			if len(embeddings[-1]) != 300:
				print(embeddings[-1])
				print(len(embeddings[-1]))
				raise ValueError('Error on reading glove embeddings from file!')

			line = f.readline()
			index += 1

	# replace id == 0 with token [PAD], and put that token at the end

	# add embedding of index 0, to the end
	embeddings.append( embeddings[0][:] )
	# also updating the word2idx and idx2word dictionaries
	word = idx2word[0]
	idx2word[ len(idx2word) ] = idx2word[0]
	word2idx[ idx2word[0] ] = len(word2idx)

	# set "[PAD]" token to be the token with id 0
	idx2word[ 0 ] = "[PAD]"
	word2idx[ "[PAD]" ] = 0
	embeddings[0] = [0] * 300

	# add "[CLS]" token at the end of the embeddings
	idx2word[ len(idx2word) ] = "[CLS]"
	word2idx[ "[CLS]" ] = len(word2idx)
	embeddings.append( [0] * 300 )

	embeddings = np.array(embeddings, dtype='float32')
	embeddings = torch.from_numpy(embeddings).float()
	pickle.dump( embeddings, open(os.path.join( "data/embeddings/glove.6B.300d.p"), 'wb'))
	pickle.dump( word2idx, open(os.path.join( "data/embeddings/glove.6B.300d_word2idx_dict.p"), 'wb'))
	pickle.dump( idx2word, open(os.path.join( "data/embeddings/glove.6B.300d_idx2word_dict.p"), 'wb'))

def load_glove_embeddings(path = "data/embeddings/glove.6B.300d.p"):
	return read_pickle(path)

def l1_loss_fn(repr_):
	return torch.abs(torch.mean(repr_))

def l0_loss(repr_):
	return (repr_ != 0).float().mean()

def l0_loss_fn(q_repr, d1_repr, d2_repr):
	""" L0 loss ( Number of non zero elements )
	"""
	# return mean batch l0 loss of qery, and docs
	concat_d = torch.cat([d1_repr, d2_repr], dim=1)
	non_zero_d = l0_loss(concat_d)
	non_zero_q = l0_loss(q_repr)
	return non_zero_d, non_zero_q


def read_csv(path, delimiter='\t'):
	return np.genfromtxt(path, delimiter=delimiter)

def read_data(path, delimiter='\t'):
	with open(path, 'r') as f:
		ids = list()
		data = list()
		for line in f.readlines():
			line_split = line.split(delimiter, 1)
			ids.append(line_split[0])
			data.append(line_split[1])
		return np.asarray(ids), np.asarray(data)


def read_triplets(path, delimiter='\t'):
	with open(path, newline='') as csvfile:
		return list(csv.reader(csvfile, delimiter=delimiter))

def read_qrels(path, delimiter='\t'):
	data = list()
	with open(path, 'r') as file:
		lines = file.readlines()
		for line in lines:
			line_split = line.strip().split(delimiter)
			data.append([line_split[0], line_split[2]])
	return data

def read_pickle(path):
	return pickle.load(open(path, 'rb'))

def write_pickle(object, path):
	pickle_out = open(path,"wb")
	pickle.dump(object, pickle_out)
	pickle_out.close()

def read_json(path):
	with open(path, "r") as read_file:
		return json.load(read_file)

def str2lst(string):
	if '-' not in string:
		return [int(string)]
	return [int(s) for s in string.split('-')]

def get_index_line_from_file(file, index_seek_dict, index):
	""" Given a seek value and a file, read the line that follows that seek value
	"""
	file.seek( index_seek_dict[index] )
	return file.readline()


def get_ids_from_tsv(line):
	delim_pos = line.find(' ')
	id_ = int(line[:delim_pos])
	ids = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
	return id_, ids



def cv_squared(x, device):
	"""The squared coefficient of variation of a sample.
	Useful as a loss to encourage a positive distribution to be more uniform.
	Epsilons added for numerical stability.
	Returns 0 for one example in batch
	Args:
	x: a `Tensor`.
	Returns:
	a `Scalar`.
	"""
	eps = 1e-10
	# if only one example in batch
	if x.shape[0] == 1:
		return torch.FloarTensor(0,device=device)
	return x.float().var() / (x.float().mean()**2 + eps)


def activations_to_load(activations):
	"""Compute the true load per latent term, given the activations.
	The load is the number of examples for which the corresponding latent term is != 0.
	Args:
	gates: a `Tensor` of shape [batch_size, hidden_dim]
	Returns:
	a float32 `Tensor` of shape [hidden_dim]
	"""
	load = (activations != 0).sum(0)
	return load


def balance_loss_fn(actications, device):
	"""
	Loss to balance the load of the latent terms
	"""
	load = activations_to_load(actications)
	return cv_squared(load, device)

def get_posting_lengths(reprs, sparse_dims):
	lengths = np.zeros(sparse_dims)
	for repr_ in  reprs:
		lengths += (repr_ != 0).sum(0)
	return lengths

def get_latent_terms_per_doc(reprs):
	terms = list()
	for repr_ in reprs:
		terms += (repr_ != 0).sum(1).tolist()
	return terms


def plot_histogram_of_latent_terms(path, reprs, name):
	sparse_dims = reprs[0].shape[1]
	latent_terms_per_doc = get_latent_terms_per_doc(reprs)
	sns.distplot(latent_terms_per_doc, bins=sparse_dims//10)
	# plot histogram
	# save histogram

	plt.ylabel('Frequency')
	# plt.xlabel('Dimension of the Representation Space (sorted)')
	plt.xlabel('# Latent Terms')
	plt.savefig(path + f'/num_latent_terms_per_{name}.pdf', bbox_inches='tight')
	plt.close()


def plot_ordered_posting_lists_lengths(path,reprs, name, n=-1):
	sparse_dims = reprs[0].shape[1]
	frequencies = get_posting_lengths(reprs, sparse_dims)
	n = n if n > 0 else len(frequencies)
	top_n = sorted(frequencies, reverse=True)[:n]
	# print(top_n)
	# run matplotlib on background, not showing the plot


	plt.plot(top_n)
	plt.ylabel('Frequency')
	# plt.xlabel('Dimension of the Representation Space (sorted)')
	n_text = f' (top {n})' if n != len(frequencies) else ''
	plt.xlabel('Latent Dimension (Sorted)' + n_text)
	plt.savefig(path+ f'/num_{name}_per_latent_term.pdf', bbox_inches='tight')
	plt.close()


def add_before_ending(filename, add_before_ending):
	name, ending = os.path.splitext(filename)
	return name + add_before_ending + ending



def write_ranking(scores, q_ids, results_file_path):

	results_file = open(results_file_path, 'w')
	for i, q_id in enumerate(q_ids):
		for j, (doc_id, score) in enumerate(scores[i]):
			results_file.write(f'{q_id}\t{doc_id}\t{j+1}\n' )

	results_file.close()



def write_ranking_trec(scores, q_ids, results_file_path):

	results_file = open(results_file_path, 'w')
	for i, q_id in enumerate(q_ids):
		for j, (doc_id, score) in enumerate(scores[i]):
			results_file.write(f'{q_id}\tQ0\t{doc_id}\t{j+1}\t{score}\teval\n')
	results_file.close()

def _getThreads():
	""" Returns the number of available threads on a posix/win based system """
	try:
		if sys.platform == 'win32':
			return (int)(os.environ['NUMBER_OF_PROCESSORS'])
		else:
			return (int)(os.popen('grep -c cores /proc/cpuinfo').read())
	except:
		return 0



def get_model_folder_name(cfg):
		if cfg.model == "tf":
			# updating hidden dimensions according to selected embeddings
			if cfg.embedding == "bert":
				cfg.tf.hidden_size=768
			elif cfg.embedding == "glove":
				cfg.tf.hidden_size=300

			model_string=f"{cfg.model.upper()}_L_{cfg.tf.num_of_layers}_H_{cfg.tf.num_attention_heads}_D_{cfg.tf.hidden_size}_P_{cfg.tf.pooling_method}_ACT_{cfg.tf.act_func}"

			if cfg.tf.layer_norm == False:
				model_string += "_no_layer_norm"
			if cfg.balance_scalar != 0:
				model_string += f'bal_{cfg.balance_scalar}'
		elif cfg.model == "snrm":
			model_string=f"{cfg.model.upper()}_n-gram_{cfg.snrm.n_gram_model}_{cfg.snrm.hidden_sizes}"

		elif cfg.model =="rank":
			trainable_weights_str = "_trainable" if cfg.rank_model.trainable_weights else ""
			weights_str = "IDF" if "idf" in cfg.rank_model.weights else cfg.rank_model.weights
			model_string=f"{cfg.model.upper()}_weights_{weights_str}{trainable_weights_str}_{cfg.rank_model.hidden_sizes}"

		elif cfg.model =="bert":

			load_bert_layers = cfg.bert.load_bert_layers
			load_bert_path = cfg.bert.load_bert_path

			if isinstance(load_bert_layers, str) and len(load_bert_layers) != 0:
				load_bert_layers = str2lst(str(load_bert_layers))
			elif isinstance(load_bert_layers, int):
				load_bert_layers = [load_bert_layers]
			else:
				load_bert_layers = []

			# if we are not using a pretrained bert
			if len(load_bert_layers) == 0:
				model_string=f"BERT_L_{cfg.tf.num_of_layers}_H_{cfg.tf.num_attention_heads}_D_{cfg.tf.hidden_size}_from_scratch"
			# if we are using a pretrained bert
			else:
				temp_load_bert_path = load_bert_path.replace("/", ".")
				model_string = "BERT_loaded_" + temp_load_bert_path + "_Layers_" + str(cfg.bert.load_bert_layers)


		else:
			raise ValueError("Model not set properly!:", cfg.model)

		if cfg.large_out_biases:
			model_string += "_large_out_biases"

		if cfg.dataset == "robust04":
			if cfg.provided_triplets:
				model_string += "_provided_triplets"
			else:
				model_string +=  '_sample_' + cfg.sampler + "_target_" + cfg.target
				if cfg.samples_per_query != -1:
					model_string += "_comb_of_" + str(cfg.samples_per_query)
				if cfg.sample_j:
					model_string += "_sample_j"
				if cfg.single_sample:
					model_string += "_single_sample"
				if cfg.sample_random == False:
					model_string += "_No_random_triplets"

		# create experiment directory name
		if cfg.model == "bert":
			return f"{cfg.dataset}_Stop_{cfg.stopwords}_bsz_{cfg.batch_size_train}_lr_{cfg.lr}_{model_string}"
		if cfg.model != "rank":
			return f"{cfg.dataset}_l1_{cfg.l1_scalar}_margin_{cfg.margin}_Emb_{cfg.embedding}_STOP_{cfg.stopwords}_Sparse_{cfg.sparse_dimensions}_bsz_{cfg.batch_size_train}_lr_{cfg.lr}_{model_string}"
		else:
			return f"{cfg.dataset}_margin_{cfg.margin}_Emb_{cfg.embedding}_STOP_{cfg.stopwords}_bsz_{cfg.batch_size_train}_lr_{cfg.lr}_{model_string}"


def plot_top_k_analysis(analysis_dict):

	MRR_top_k_freq = analysis_dict["MRR_top_k_freq"]
	MRR_bottom_k_freq = analysis_dict["MRR_bottom_k_freq"]
	MRR_top_k_var = analysis_dict["MRR_top_k_var"]
	MRR_bottom_k_var = analysis_dict["MRR_bottom_k_var"]
	most_freq_dims = analysis_dict["most_freq_dims"]
	least_freq_dims = analysis_dict["least_freq_dims"]
	most_var_dims = analysis_dict["most_var_dims"]
	least_var_dims = analysis_dict["least_var_dims"]



	x = np.arange(len(MRR_top_k_freq))

	fig, axs = plt.subplots(2)
	fig.suptitle('Top and Bottom k analysis (Ignoring dimensions one-by-one)')
	axs[0].plot(x, MRR_top_k_freq, "-b", label="Frequent")
	axs[0].plot(x, MRR_top_k_var, "-r", label="Variant")
	axs[0].legend(loc="upper right")
	axs[0].axis(ymin=0, ymax=1.0)
	axs[0].title.set_text('Ignoring TOP k')

	axs[1].plot(x, MRR_bottom_k_freq, "-b", label="Frequent")
	axs[1].plot(x, MRR_bottom_k_var, "-r", label="Variant")
	axs[1].legend(loc="upper right")
	axs[1].axis(ymin=0, ymax=1.0)
	axs[1].title.set_text('Ignoring BOTTOM k')

	plt.show()


def create_random_batch(bsz, max_len, vocab_size):
	input = torch.randint(0, vocab_size, (bsz * 3, max_len))

	lengths = torch.tensor([max_len - 1 for i in range(bsz * 3)])


	targets = torch.randint(0, 2, (bsz,1))

	return input, lengths, targets

def create_random_batch_bert_interaction(bsz, max_len, vocab_size):

	targets = (torch.randn(bsz) > 0.5).long()

	input_ids = torch.randint(0, vocab_size, (bsz , max_len))
	attention_masks = torch.randint(0,2, (bsz , max_len)).bool()
	token_type_ids = torch.randint(0,2, (bsz , max_len))

	return input_ids, attention_masks, token_type_ids, targets

def get_max_samples_per_gpu(model, device, n_gpu, optim, loss_fn, max_len, vocab_size):

	bsz = 1

	if isinstance(model, torch.nn.DataParallel):
		model_type = model.module.model_type
	else:
		model_type = model.model_type

	try:
		while True :

			if model_type == "bert-interaction":

				input_ids, attention_masks, token_type_ids, targets = create_random_batch_bert_interaction(bsz * n_gpu, max_len, vocab_size=vocab_size)
				input_ids, attention_masks, token_type_ids, targets = input_ids.to(device), attention_masks.to(device), token_type_ids.to(device), targets.to(device)

			else:
				data, lengths, targets = create_random_batch(bsz * n_gpu, max_len, vocab_size)
				data, lengths, targets = data.to(device), lengths.to(device), targets.to(device)

			if model_type == "representation-based":
				# forward pass (inputs are concatenated in the form [q1, q2, ..., q1d1, q2d1, ..., q1d2, q2d2, ...])
				logits = model(data, lengths)

				# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
				split_size = logits.size(0) // 3
				q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

				# performing inner products
				score_q_d1 = torch.bmm(q_repr.unsqueeze(1), d1_repr.unsqueeze(-1)).squeeze()
				score_q_d2 = torch.bmm(q_repr.unsqueeze(1), d2_repr.unsqueeze(-1)).squeeze()

				# if batch contains only one sample the dotproduct is a scalar rather than a list of tensors
				# so we need to unsqueeze
				if bsz == 1:
					score_q_d1 = score_q_d1.unsqueeze(0)
					score_q_d2 = score_q_d2.unsqueeze(0)

				# calculate l1 loss
				l1_loss = l1_loss_fn(torch.cat([q_repr, d1_repr, d2_repr], 1))
				# calculate balance loss
				balance_loss = balance_loss_fn(logits, device)
				# calculating L0 loss
				l0_q, l0_docs = l0_loss_fn(d1_repr, d2_repr, q_repr)

			elif model_type == "interaction-based":

			# # if the model provides a score for a document and a query
			# elif isinstance(model, RankModel):
								# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
				split_size = data.size(0) // 3
				q_repr, doc1, doc2 = torch.split(data, split_size)
				lengths_q, lengths_d1, lengths_d2 = torch.split(lengths, split_size)
				score_q_d1 = model(q_repr, doc1, lengths_q, lengths_d1)
				score_q_d2 = model(q_repr, doc2, lengths_q, lengths_d2)

				# calculate l1 loss
				l1_loss = 0
				# calculate balance loss
				balance_loss = 0
				# calculating L0 loss
				l0_q, l0_docs = 0, 0

			elif model_type == "bert-interaction":

				output = model(input_ids, attention_masks, token_type_ids)

			else:
				raise ValueError(f"run_model.py , model_type not properly defined!: {model_type}")


				# calculating loss
			if model_type == "bert-interaction":
				loss = loss_fn(output, targets)
			else:
				loss = loss_fn(score_q_d1, score_q_d2, targets)

			loss.backward()
			optim.step()
			optim.zero_grad()

			bsz *= 2


	except RuntimeError as e:
		if 'out of memory' in str(e):

			# print("Dynamically calculated max_samples_per_gpu == ", bsz - 1)
			optim.zero_grad()
			torch.cuda.empty_cache()

			max_samples_per_gpu = bsz - 2

			return max_samples_per_gpu

		else:
			raise e





def load_model(cfg, load_model_folder, device):
	cfg.embedding = 'random'
	model, device, n_gpu, _ = instantiate_model(cfg)
	state_dict = torch.load(load_model_folder + '/best_model.model', map_location=device)

	if not isinstance(state_dict, collections.OrderedDict):

		if isinstance(state_dict, torch.nn.DataParallel):
			state_dict = state_dict.module
		state_dict = state_dict.state_dict()

	new_state_dict = collections.OrderedDict()
	for k, v in state_dict.items():

		if n_gpu < 2:
			if 'module.' in k:
				k = k.replace('module.', '')
		else:
			if 'module.' not in k:
				k = 'module.' + k
		new_state_dict[k] = v
	state_dict = new_state_dict

	model.load_state_dict(state_dict)
	return model
