
import torch
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

def file_len(fname):
	""" Get the number of lines from file
	""" # https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
	p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	result, err = p.communicate()
	if p.returncode != 0:
		raise IOError(err)
	return int(result.strip().split()[0])



def collate_fn_padd(batch):
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
		q, doc1, doc2 = item[0]
		batch_q.append(torch.ShortTensor(q))
		batch_doc1.append(torch.ShortTensor(doc1))
		batch_doc2.append(torch.ShortTensor(doc2))
		batch_targets.append(item[1])
	batch_data = batch_q + batch_doc1 + batch_doc2
	batch_targets = torch.Tensor(batch_targets)

	batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
	#padd data along axis 1
	batch_data = pad_sequence(batch_data,1).long()
	return batch_data, batch_targets, batch_lengths

# def padd_sentences()



def get_pretrained_BERT_embeddings():
	bert = transformers.BertModel.from_pretrained('bert-base-uncased')
	return bert.embeddings.word_embeddings.weight



def load_glove_embeddings(path):
	""" Load Glove embeddings from file
	"""
	embeddings = []
	with open(path) as f:
		index = 0
		line = f.readline()
		while(line):
			line = line.split()
			word = line[0]
			embeddings.append( [ line[1:] ] )
			line = f.readline()
			index += 1
	embeddings = np.array(embeddings, dtype='float32')
	embeddings = torch.from_numpy(embeddings).float()
	return embeddings

	# with open(path) as f:
	# 	embeddings = np.zeros((len(word2idx), embedding_dim))
	# 	for index, line in enumerate(f.readlines()):
	# 		values = line.split()
	# 		word = values[0]
	# 		index = word2idx.get(word)
	# 		if index:
	# 			vector = np.array(values[1:], dtype='float32')
	# 			embeddings[index] = vector
	# 	return torch.from_numpy(embeddings).float().to(device)

def generate_word2idx_dict_from_glove(path):
	word2idx = {}
	with open(path) as f:
		line_counter = 0
		line = f.readline()
		while(line):
			word = line.split()[0]
			word_id = line_counter
			word2idx[word] = word_id
			line = f.readline()
			line_counter += 1
	pickle.dump( word2idx, open(os.path.join( path +  'word2idx_dict.p'), 'wb'))


def l1_loss_fn(q_repr, d1_repr, d2_repr):
	""" L1 loss ( Sum of vectors )
	"""
	concat = torch.cat([q_repr, d1_repr, d2_repr], 1)
	return torch.mean(torch.sum(concat, 1))/q_repr.size(1)


def l0_loss_fn(q_repr, d1_repr, d2_repr):
	""" L0 loss ( Number of non zero elements )
	"""
	# return mean batch l0 loss of qery, and docs
	concat_d = torch.cat([d1_repr, d2_repr], dim=1)
	non_zero_d = (concat_d > 0).float().mean(1).mean()
	non_zero_q = (q_repr > 0).float().mean(1).mean()
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

def read_json(path):
	with open(path, "r") as read_file:
		return json.load(read_file)

def str2lst(string):
	if '-' not in string:
		return [int(string)]
	return [int(s) for s in string.split('-')]


def create_seek_dictionary_per_index(filename, delimiter=' ', line_index_is_id = True):
	""" Creating a dictionary, for accessing directly a documents content, given document's id
			from a large file containing all documents
			returns:
			dictionary [doc_id] -> Seek value of a large file, so that you only have to read the exact document (doc_id)
	"""
	index_to_seek = {}
	sample_counter = 0

	with open(filename) as file:

		seek_value = file.tell()
		line = file.readline()
		while line:
			split_line = line.strip().split(delimiter)
			# triplets so use counter as id
			if line_index_is_id:
				id_ = sample_counter
			else:
				id_ = split_line[0]
			sample_counter += 1
			index_to_seek[id_] = seek_value
			if sample_counter % 100000 == 0:
				print(sample_counter)
			seek_value = file.tell()
			line = file.readline()

	return index_to_seek

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
    The load is the number of examples for which the corresponding latent term is >0.
    Args:
    gates: a `Tensor` of shape [batch_size, hidden_dim]
    Returns:
    a float32 `Tensor` of shape [hidden_dim]
    """
    load = (activations > 0).sum(0)
    return load


def balance_loss_fn(actications, device):
    """
    Loss to balance the load of the latent terms
    """
    load = activations_to_load(actications)
    return cv_squared(load, device)


def plot_histogram_of_latent_terms(path, latent_terms_per_doc, vocab_size):

	sns.distplot(latent_terms_per_doc, bins=vocab_size//10)
	# plot histogram
	# save histogram

	plt.ylabel('Document Frequency')
	# plt.xlabel('Dimension of the Representation Space (sorted)')
	plt.xlabel('# Latent Terms')
	plt.savefig(path + '/num_latent_terms_per_doc.pdf', bbox_inches='tight')
	plt.close()


def plot_ordered_posting_lists_lengths(path,frequencies, n=-1):
	n = n if n > 0 else len(frequencies)
	top_n = sorted(frequencies, reverse=True)[:n]
	# print(top_n)
	# run matplotlib on background, not showing the plot


	plt.plot(top_n)
	plt.ylabel('Document Frequency')
	# plt.xlabel('Dimension of the Representation Space (sorted)')
	n_text = f' (top {n})' if n != len(frequencies) else ''
	plt.xlabel('Latent Dimension (Sorted)' + n_text)
	plt.savefig(path+ '/num_docs_per_latent_term.pdf', bbox_inches='tight')
	plt.close()
