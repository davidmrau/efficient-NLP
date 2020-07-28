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
		#if patience == 0:
		#	 self.is_better = lambda a, b: True
		#	 self.step = lambda a: False

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


def padd_tensor(sequences, max_len):
    """
    :param sequences: list of tensors
    :return:
    """
    num = len(sequences)
    out_dims = (num, max_len, *sequences[0].shape[1:])
    out_tensor = sequences[0].data.new(*out_dims).fill_(0)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length] = tensor
    return out_tensor


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



def gen_folds(dataset_len, num_folds):
	folds = list()
	rand_indices = list(range(dataset_len))
	random.shuffle(rand_indices)
	for i in range(1,num_folds+1):
		# train the model
		from_ = dataset_len*(i-1)//num_folds
		to_ = int(np.floor(dataset_len*i/num_folds))
		test_indices = rand_indices[from_:to_]
		train_indices = rand_indices[:from_] + rand_indices[to_:]
		folds.append([train_indices, test_indices])
	return folds

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

	# the model might be "tf" or "bert". We need to handle each case dynamically
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
			model = torch.load(load_bert_path)

		model_state_dict = model.state_dict()

		# update the number of layers, depending on the layers that need to be copied
		cfg.__getattr__(model_name)["num_of_layers"] =  max( max(load_bert_layers) -1, cfg.__getattr__(model_name)["num_of_layers"])

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

	elif cfg.model == "rank":
		model = RankModel(hidden_sizes = str2lst(str(cfg.rank_model.hidden_sizes)), embedding_parameters = embedding_parameters,
			embedding_dim = cfg.rank_model.embedding_dim, vocab_size = cfg.vocab_size, dropout_p = cfg.rank_model.dropout_p,
			weights = cfg.rank_model.weights, trainable_weights = cfg.rank_model.trainable_weights)

	elif cfg.model == "bert":
		params_to_copy = utilize_pretrained_bert(cfg)

		model = BERT_inter(hidden_size = cfg.bert.hidden_size, num_of_layers = cfg.bert.num_of_layers,
							num_attention_heads = cfg.bert.num_attention_heads, input_length_limit = cfg.bert.input_length_limit,
							vocab_size = cfg.vocab_size, embedding_parameters = embedding_parameters, params_to_copy = params_to_copy,
							point_wise = cfg.bert.point_wise)

	elif cfg.model == "tf":
		params_to_copy = utilize_pretrained_bert(cfg)

		model = BERT_based( hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
							sparse_dimensions = cfg.sparse_dimensions, num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = cfg.tf.input_length_limit,
							vocab_size = cfg.vocab_size, embedding_parameters = embedding_parameters, pooling_method = cfg.tf.pooling_method,
							large_out_biases = cfg.large_out_biases, layer_norm = cfg.tf.layer_norm, act_func = cfg.tf.act_func,
							params_to_copy = params_to_copy)

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

	return model, device, n_gpu

def collate_fn_padd_single(batch):
	""" Collate function for aggregating samples into batch size.
		returns:
		batch_data = Torch([ id_1, tokens_2, ..., id_2, tokens_2, ... ])
		batch_lengths = length of each query/document,
			that is used for proper averaging ignoring 0 padded inputs
	"""
	#batch * [id, tokens]

	batch_lengths = list()
	batch_ids, batch_data = list(), list()

	for item in batch:
		# for weak supervision datasets, some queries/documents have empty text.
		# In that case the sample is None, and we skip this samples
		if item is None:
			continue

		id_, tokens = item
		batch_data.append(torch.IntTensor(tokens))
		batch_ids.append(id_)

	# in case this batch does not contain any samples, then we return None
	if len(batch_data) == 0:
		return None

	batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
	#pad data along axis 1
	batch_data = pad_sequence(batch_data,1).long()
	return batch_ids, batch_data, batch_lengths


def collate_fn_padd_triples(batch):
	""" Collate function for aggregating samples into batch size.
		returns:
		batch_data = Torch([ q_1, q_2, ..., q1_d1, q2_d1, ..., q2_d1, q2_d2, ... ])
		batch_targets = -1 or 1 for each sample
		batch_lenghts = lenght of each query/document,
			that is used for proper averaging ignoring 0 padded inputs
	"""

	#batch * [q, d1, d2], target

	batch_lengths = list()
	batch_targets = list()
	batch_q, batch_doc1, batch_doc2 = list(), list(), list()

	for item in batch:
		# for weak supervision datasets, some queries/documents have empty text.
		# In that case the sample is None, and we skip these samples
		if item is None:
			continue


		q, doc1, doc2 = item[0]
		batch_q.append(torch.IntTensor(q))
		batch_doc1.append(torch.IntTensor(doc1))
		batch_doc2.append(torch.IntTensor(doc2))
		batch_targets.append(item[1])

	batch_data = batch_q + batch_doc1 + batch_doc2
	# in case this batch does not contain any samples, then we return None
	if len(batch_data) == 0:
		return None

	batch_targets = torch.Tensor(batch_targets)

	batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
	#pad data along axis 1
	batch_data = pad_sequence(batch_data,1).long()
	return batch_data, batch_targets, batch_lengths


def split_batch_to_minibatches_bert_interaction(batch, max_samples_per_gpu = 2, n_gpu = 1, pairwise_training = False):

	if pairwise_training:
		max_samples_per_gpu *= 2

	# calculate the number of minibatches so that the maximum number of samples per gpu is maintained
	size_of_minibatch = max_samples_per_gpu * n_gpu

	batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_targets = batch

	number_of_samples_in_batch = batch_input_ids.size(0)

	if number_of_samples_in_batch <= max_samples_per_gpu:

		if pairwise_training:
			# rows represent sample pairs, column 1 and 2 represent target of d1 and d2 respecttively
			batch_targets = batch_targets.view(-1,2)
			# rows represent sample pairs, 1 means d1 is more relevant, -1 means d2 is more relevant
			batch_targets = (batch_targets[:,0] > batch_targets[:,1]).int()*2 -1

			batch = [batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_targets]

		return [batch]

	number_of_minibatches = math.ceil(number_of_samples_in_batch / size_of_minibatch)

	minibatches = []

	if pairwise_training == False:

		for i in range(number_of_minibatches):
			minibatches.append([ batch_input_ids[i*size_of_minibatch : (i+1)*size_of_minibatch], \
				batch_attention_masks[i*size_of_minibatch : (i+1)*size_of_minibatch],  \
				batch_token_type_ids[i*size_of_minibatch : (i+1)*size_of_minibatch],  \
				batch_targets[i*size_of_minibatch : (i+1)*size_of_minibatch] ])

	else:

		# preprocess targets so that they prepresent pair targets

		# rows represent sample pairs, column 1 and 2 represent target of d1 and d2 respecttively
		batch_targets = batch_targets.view(-1,2)
		# rows represent sample pairs, 1 means d1 is more relevant, -1 means d2 is more relevant
		batch_targets = (batch_targets[:,0] > batch_targets[:,1]).int()*2 -1
		# in a pairwise training case, the number of targets are half of the size of the input samples
		size_of_minibatch_targets = int(size_of_minibatch/2)

		for i in range(number_of_minibatches):
			minibatches.append([ batch_input_ids[i*size_of_minibatch : (i+1)*size_of_minibatch], \
				batch_attention_masks[i*size_of_minibatch : (i+1)*size_of_minibatch],  \
				batch_token_type_ids[i*size_of_minibatch : (i+1)*size_of_minibatch],  \
				batch_targets[i*size_of_minibatch_targets : (i+1)*size_of_minibatch_targets] ])

	return minibatches

def create_bert_inretaction_input(q, doc):
	# inputs: query and document token ids in the form of numpy array
	# creates one input in the form : ["CLS"_id, q_t1_id, ..., q_tN_id, "SEP"_id, d_t1_id, ..., d_tM_id]
	# and the return the above input_ids, along with the corresponding attention mask and token type ids

	# Hard-coding token IDS of CLS, SEP and PAD
	cls_token_id = np.array([101])
	sep_token_id = np.array([102])
	pad_token_id = 0

	np_zero = np.array([0])

	q_token_type_ids = np.zeros(q.shape[0])
	d_token_type_ids = np.ones(doc.shape[0])

	q_d = np.concatenate([cls_token_id, q, sep_token_id, doc])
	q_d = torch.LongTensor(q_d)

	q_d_attention_mask = torch.ones(q_d.shape[0], dtype=torch.bool)

	q_d_token_type_ids = np.concatenate([np_zero, q_token_type_ids, np_zero, d_token_type_ids])
	q_d_token_type_ids = torch.LongTensor(q_d_token_type_ids)

	return q_d, q_d_attention_mask, q_d_token_type_ids


def collate_fn_bert_interaction(batch):
	""" Collate function for aggregating samples into batch size.
		returns:
		batch_input_ids = Torch.int32([ 
						["CLS"_id, q1_t1_id, ..., q1_tN_id, "SEP"_id, q1_d1_t1_id, ..., q1_d1_tM_id, "PAD", ... ],
						["CLS"_id, q1_t1_id, ..., q1_tN_id, "SEP"_id, q1_d2_t1_id, ..., q1_d2_tM_id, "PAD", ... ], ... ])
						size : BSZ X MAX_batch_length
						The first "PAD" at the beginning will be replaced with "CLS" during the forward pass of the model, using batch_pad_mask
						The second "PAD" between the query and the document will be replaced with "SEP" during the forward pass of the model, using batch_sep_mask

		batch_attention_masks = bollean where non padded tokens are True, size equal with batch_input_ids : BSZ X MAX_batch_length

		batch_attention_masks = mask that defines if the token is from query (value = 0) or document (value = 1),
								size equal with batch_input_ids : BSZ X MAX_batch_length

		batch_targets = [q1_d1_sim_score, q1_d2_sim_score, ...], scores are in {-1,1}

	"""


	batch_input_ids =list()
	batch_attention_masks =list()
	batch_token_type_ids = list()
	batch_targets =list()

	for item in batch:
		# for weak supervision datasets, some queries/documents have empty text.
		# In that case the sample is None, and we skip these samples
		if item is None:
			continue

		q, doc1, doc2 = item[0]

		target = item[1]

		if target == 1:
			q_d1_target = 1
			q_d2_target = 0
		else:
			q_d1_target = 0
			q_d2_target = 1


		# if random.random() > 0.5:
		# 	return [query, doc1, doc2], 1
		# else:
		# 	return [query, doc2, doc1], -1


		q_d1, q_d1_attention_mask, q_d1_token_type_ids = create_bert_inretaction_input(q, doc1)
		q_d2, q_d2_attention_mask, q_d2_token_type_ids = create_bert_inretaction_input(q, doc2)

		batch_input_ids.append(q_d1)
		batch_input_ids.append(q_d2)

		batch_attention_masks.append(q_d1_attention_mask)
		batch_attention_masks.append(q_d2_attention_mask)

		batch_token_type_ids.append(q_d1_token_type_ids)
		batch_token_type_ids.append(q_d2_token_type_ids)


		batch_targets.append(q_d1_target)
		batch_targets.append(q_d2_target)

	# in case this batch does not contain any samples, then we return None
	if len(batch_input_ids) == 0:
		return None

	batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
	batch_attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=False)
	batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True, padding_value=0)
	batch_targets = torch.LongTensor(batch_targets)

	return batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_targets



def split_batch_to_minibatches(batch, max_samples_per_gpu = 2, n_gpu = 1):

	if max_samples_per_gpu == -1:
		return [batch]

	data, targets, lengths = batch

	# calculate the number of minibatches so that the maximum number of samples per gpu is maintained
	size_of_minibatch = max_samples_per_gpu * n_gpu

	split_size = data.size(0) // 3
	queries, doc1, doc2 = torch.split(data, split_size)
	queries_len, doc1_len, doc2_len = torch.split(lengths, split_size)

	number_of_samples_in_batch = queries.size(0)

	if number_of_samples_in_batch <= max_samples_per_gpu:
		return [batch]


	number_of_minibatches = math.ceil(number_of_samples_in_batch / size_of_minibatch)

	# arrange the minibatches
	minibatches = []
	for i in range(number_of_minibatches):

		minibatch_queries =  queries[ i * size_of_minibatch : (i+1) * size_of_minibatch ]

		minibatch_d1 =  doc1[  i * size_of_minibatch : (i+1) * size_of_minibatch ]
		minibatch_d2 =  doc2[  i * size_of_minibatch : (i+1) * size_of_minibatch ]

		minibatch_queries_lengths =  queries_len[ i * size_of_minibatch : (i+1) * size_of_minibatch ]

		minibatch_d1_lengths =  doc1_len[  i * size_of_minibatch : (i+1) * size_of_minibatch ]
		minibatch_d2_lengths =  doc2_len[  i * size_of_minibatch : (i+1) * size_of_minibatch ]

		minibatch_targets =  targets[ i * size_of_minibatch : (i+1) * size_of_minibatch ]

		minibatch_data = torch.cat([minibatch_queries , minibatch_d1 , minibatch_d2], dim = 0)
		minibatch_lengths = torch.cat([minibatch_queries_lengths , minibatch_d1_lengths , minibatch_d2_lengths], dim = 0)

		minibatch = [minibatch_data, minibatch_targets, minibatch_lengths]

		minibatches.append(minibatch)
	return minibatches


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
			results_file.write(f'{q_id}\t0\t{doc_id}\t{j+1}\t{score}\teval\n')
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
				model_string = "BERT_loaded_" + temp_load_bert_path + "_Layers_" + cfg.bert.load_bert_layers

			if cfg.bert.point_wise:
				model_string = "POINT_wise_" + model_string
			else:
				model_string = "PAIR_wise_" + model_string
		
		else:
			raise ValueError("Model not set properly!:", cfg.model)

		if cfg.large_out_biases:
			model_string += "_large_out_biases"

		if cfg.dataset == "robust04":
			model_string +=  '_sample_' + cfg.sampler + "_target_" + cfg.target
			if cfg.samples_per_query != -1:
				model_string += "_comb_of_" + str(cfg.samples_per_query)
			if cfg.sample_j:
				model_string += "_sample_j"
			if cfg.single_sample:
				model_string += "_single_sample"

		# create experiment directory name
		if cfg.model == "bert":
			return f"{cfg.dataset}_bsz_{cfg.batch_size_train}_lr_{cfg.lr}_{model_string}"
		if cfg.model != "rank":
			return f"{cfg.dataset}_l1_{cfg.l1_scalar}_margin_{cfg.margin}_Emb_{cfg.embedding}_Sparse_{cfg.sparse_dimensions}_bsz_{cfg.batch_size_train}_lr_{cfg.lr}_{model_string}"
		else:
			return f"{cfg.dataset}_margin_{cfg.margin}_Emb_{cfg.embedding}_bsz_{cfg.batch_size_train}_lr_{cfg.lr}_{model_string}"


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

	# fig = plt.figure()
	# ax1 = fig.add_subplot(121)
	# ax2 = fig.add_subplot(122)

	# axs = [ax1, ax2]


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

	# plt.title('Removing the top k dimensions one-by-one')

	# axs[1].plot(x, -y)



	plt.show()




def create_random_batch(bsz, max_len):
	input = torch.randint(0, 2, (bsz * 3, max_len))

	lengths = torch.tensor([max_len - 1 for i in range(bsz * 3)])


	targets = torch.randint(0, 2, (bsz,1))

	return input, lengths, targets

def create_random_batch_bert_interaction(bsz, max_len, point_wise):

	if point_wise:
		targets = (torch.randn(bsz) > 0.5).long()
	else:
		targets = (torch.randn(bsz) > 0.5).int()*2 -1
		bsz *= 2


	input_ids = torch.randint(0,2, (bsz , max_len))
	attention_masks = torch.randint(0,2, (bsz , max_len)).bool()
	token_type_ids = torch.randint(0,2, (bsz , max_len))

	return input_ids, attention_masks, token_type_ids, targets

def get_max_samples_per_gpu(model, device, n_gpu, optim, loss_fn, max_len):

	bsz = 2

	# max_len = 100

	if isinstance(model, torch.nn.DataParallel):
		model_type = model.module.model_type
	else:
		model_type = model.model_type

	try:
		while True :

			if model_type == "bert-interaction" or model_type == "bert-interaction_pair_wise":
				point_wise = model_type == "bert-interaction"

				input_ids, attention_masks, token_type_ids, targets = create_random_batch_bert_interaction(bsz * n_gpu, max_len, point_wise = point_wise)
				input_ids, attention_masks, token_type_ids, targets = input_ids.to(device), attention_masks.to(device), token_type_ids.to(device), targets.to(device)

			else:
				data, lengths, targets = create_random_batch(bsz * n_gpu, max_len)
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

			elif model_type == "bert-interaction_pair_wise":

				num_of_samples = int(input_ids.size(0) / 2)

				# every second sample corresponds to d2, so we split inputs accordingly
				input_ids = input_ids.view(num_of_samples,2, -1)
				attention_masks = attention_masks.view(num_of_samples,2,-1)
				token_type_ids = token_type_ids.view(num_of_samples,2,-1)

				# print("init")
				# print(input_ids.size())
				# print(attention_masks.size())
				# print(token_type_ids.size())
				# # exit()

				qd1_input_ids = input_ids[:,0]
				qd2_input_ids = input_ids[:,1]

				qd1_attention_masks = attention_masks[:,0]
				qd2_attention_masks = attention_masks[:,1]

				qd1_token_type_ids = token_type_ids[:,0]
				qd2_token_type_ids = token_type_ids[:,1]

				# print("qd1 sizes")
				# print(qd1_input_ids.size())
				# print(qd1_attention_masks.size())
				# print(qd1_token_type_ids.size())
				# # print(.size())
				# # exit()

				score_q_d1 = model(qd1_input_ids, qd1_attention_masks, qd1_token_type_ids)
				score_q_d2 = model(qd2_input_ids, qd2_attention_masks, qd2_token_type_ids)

				score_q_d1 = torch.sigmoid(score_q_d1)
				score_q_d2 = torch.sigmoid(score_q_d2)

				# print("output sizes")
				# print(score_q_d1.size())
				# print(score_q_d2.size())
				# print(targets.size())

				# print(bsz, input_ids.size())


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

			bsz += 1


	except RuntimeError as e:
		if 'out of memory' in str(e):

			# print("Dynamically calculated max_samples_per_gpu == ", bsz - 1)
			optim.zero_grad()
			torch.cuda.empty_cache()
			return bsz - 3

		else:
			raise e





def load_model(cfg, load_model_folder, device, state_dict=False):
	cfg.embedding = 'random'
	model, device, n_gpu = instantiate_model(cfg)
	if not state_dict:
		model_old = torch.load(load_model_folder + '/best_model.model', map_location=device)

		if isinstance(model_old, torch.nn.DataParallel):
			model_old = model_old.module

		state_dict = model_old.state_dict()
		model.load_state_dict(state_dict)
	else:
		model.load_state_dict(torch.load(load_model_folder + '/best_model_state_dict.model'))

	return model
