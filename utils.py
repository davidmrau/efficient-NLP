
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import json
import pickle
import csv
import subprocess
import transformers
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
import math
import sys
from bert_based import BERT_based
from snrm import SNRM

def file_len(fname):
	""" Get the number of lines from file
	""" # https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
	p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	result, err = p.communicate()
	if p.returncode != 0:
		raise IOError(err)
	return int(result.strip().split()[0])


def instantiate_model(cfg):

	print('Initializing model...')

	if cfg.embedding == 'glove':
		embedding_parameters =  load_glove_embeddings(cfg.glove_embedding_path)

	elif cfg.embedding == 'bert':
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

	elif cfg.model == "tf":
		model = BERT_based( hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
							sparse_dimensions = cfg.sparse_dimensions, num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = cfg.tf.input_length_limit,
							vocab_size = cfg.vocab_size, embedding_parameters = embedding_parameters, pooling_method = cfg.tf.pooling_method,
							large_out_biases = cfg.large_out_biases, last_layer_norm = cfg.tf.last_layer_norm, act_func = cfg.tf.act_func)

	# select device depending on availability and user's setting
	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# move model to device
	model = model.to(device=device)
	if torch.cuda.device_count() > 1:
		print("Using", torch.cuda.device_count(), "GPUs!")
		model = nn.DataParallel(model)

	return model, device

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
		# In that case the sample is None, and we skip this samples
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
	return torch.mean(repr_)

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

def get_posting_lengths(reprs, sparse_dims):
	lengths = np.zeros(sparse_dims)
	for repr_ in  reprs:
		lengths += (repr_ != 0).sum(0).detach().cpu().numpy()
	return lengths

def get_latent_terms_per_doc(reprs):
	terms = list()
	for repr_ in reprs:
		terms += list((repr_ != 0).sum(1).detach().cpu().numpy())
	return terms





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


def plot_histogram_of_latent_terms(path, reprs, vocab_size, name):
	sparse_dims = reprs[0].size(1)
	latent_terms_per_doc = get_latent_terms_per_doc(reprs)
	sns.distplot(latent_terms_per_doc, bins=sparse_dims//10)
	# plot histogram
	# save histogram

	plt.ylabel('Document Frequency')
	# plt.xlabel('Dimension of the Representation Space (sorted)')
	plt.xlabel('# Latent Terms')
	plt.savefig(path + f'/num_latent_terms_per_doc_{name}.pdf', bbox_inches='tight')
	plt.close()


def plot_ordered_posting_lists_lengths(path,reprs, name, n=-1):
	sparse_dims = reprs[0].size(1)
	frequencies = get_posting_lengths(reprs, sparse_dims)
	n = n if n > 0 else len(frequencies)
	top_n = sorted(frequencies, reverse=True)[:n]
	# print(top_n)
	# run matplotlib on background, not showing the plot


	plt.plot(top_n)
	plt.ylabel('Document Frequency')
	# plt.xlabel('Dimension of the Representation Space (sorted)')
	n_text = f' (top {n})' if n != len(frequencies) else ''
	plt.xlabel('Latent Dimension (Sorted)' + n_text)
	plt.savefig(path+ f'/num_docs_per_latent_term_{name}.pdf', bbox_inches='tight')
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
    if sys.platform == 'win32':
        return (int)(os.environ['NUMBER_OF_PROCESSORS'])
    else:
        return (int)(os.popen('grep -c cores /proc/cpuinfo').read())


def get_model_folder_name(cfg):
		if cfg.model == "tf":
			# updating hidden dimensions according to selected embeddings
			if cfg.embedding == "bert":
				cfg.tf.hidden_size=768
			elif cfg.embedding == "glove":
				cfg.tf.hidden_size=300

			model_string=f"{cfg.model.upper()}_L_{cfg.tf.num_of_layers}_H_{cfg.tf.num_attention_heads}_D_{cfg.tf.hidden_size}_P_{cfg.tf.pooling_method}_ACT_{cfg.tf.act_func}"

			if cfg.tf.last_layer_norm == False:
				model_string += "_no_last_layer_norm"
			if cfg.balance_scalar != 0:
				model_string += f'bal_{cfg.balance_scalar}'
		elif cfg.model == "snrm":
			model_string=f"{cfg.model.upper()}_n-gram_{cfg.snrm.n_gram_model}_{cfg.snrm.hidden_sizes}"

		else:
			raise ValueError("Model not set properly!:", cfg.model)

		if cfg.large_out_biases:
			model_string += "_large_out_biases"

		if cfg.dataset == "robust04":
			model_string +=  '_sample_' + cfg.sampler + "_target_" + cfg.target
		# create experiment directory name
		return f"{cfg.dataset}_l1_{cfg.l1_scalar}_Emb_{cfg.embedding}_Sparse_{cfg.sparse_dimensions}_bsz_{cfg.batch_size}_lr_{cfg.lr}_{model_string}"


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
