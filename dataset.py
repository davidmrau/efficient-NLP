import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path
from fake_data import *
from random import randint
from file_interface import FileInterface
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd, add_before_ending

import pickle
import random
from tokenizer import Tokenizer


class MSMarcoTrain(data.Dataset):

	def __init__(self, triplets_path, documents_path, queries_path):

		# "open" triplets file
		self.triplets = FileInterface(triplets_path)

		# "open" documents file
		self.documents = FileInterface(documents_path)
		# "open" queries file
		self.queries = FileInterface(queries_path)


	def __len__(self):
		return len(self.triplets)

	def __getitem__(self, index):

		q_id, d1_id, d2_id = self.triplets.get_triplet(index)

		query = self.queries.get_tokenized_element(q_id)
		doc1 = self.documents.get_tokenized_element(d1_id)
		doc2 = self.documents.get_tokenized_element(d2_id)

		if np.random.random() > 0.5:
			return [query, doc1, doc2], 1
		else:
			return [query, doc2, doc1], -1


class MSMarcoSequential:
	def __init__(self, fname, batch_size):

		# open file
		self.batch_size = batch_size
		self.file_ = open(fname, 'r')

	def reset(self):
		self.file_.seek(0)

	def batch_generator(self):

		line = self.file_.readline()
		# line = self.file.readline()
		while line:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while(line and ( len(batch_ids) < self.batch_size) ):

				# getting position of '\t' that separates the doc_id and the begining of the token ids
				delim_pos = line.find('\t')
				# extracting the id
				id_ = line[:delim_pos]
				# extracting the token_ids and creating a numpy array
				tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
				batch_ids.append(id_)
				batch_data.append(torch.IntTensor(tokens_list))
				line = self.file_.readline()

			batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
			#padd data along axis 1
			batch_data = pad_sequence(batch_data,1).long()

			yield batch_ids, batch_data, batch_lengths



class MSMarcoSequentialDev:
	def __init__(self, fname, batch_size, word2index_path, embedding, is_query, min_len=5, max_len=150):

		# open file
		self.batch_size = batch_size
		self.file_ = open(fname, 'r')
		self.is_query = is_query
		self.min_len = min_len

		self.tokenizer = Tokenizer(tokenizer = embedding, max_len = max_len, stopwords='none', remove_unk = False,
							word2index_path = word2index_path, unk_words_filename = None)

		self.stop = False

	def reset(self):
		self.file_.seek(0)
		return self
		
	def tokenize(self, text):
		tokenized_ids = self.tokenizer.encode(text)

		if len(tokenized_ids) < self.min_len:
			tokenized_ids += [0]*(self.min_len - len(tokenized_ids))

		return torch.IntTensor(tokenized_ids)


	def get_id(self, line, is_query):
		spl = line.split('\t')

		if is_query:
			return spl[0]
		else:
			return spl[1]

	def get_text(self, line):
		spl = line.split('\t')

		if self.is_query:
			return spl[2]
		else:
			return spl[3]

	def batch_generator(self):

		line = self.file_.readline()
		# line = self.file.readline()
		#
		prev_q_id = self.get_id(line, is_query=True)
		curr_q_id = prev_q_id
		self.stop = False

		while line and not self.stop:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while (line and ( len(batch_ids) <= self.batch_size) ):

				id_ = self.get_id(line, self.is_query)
				curr_q_id = self.get_id(line, is_query=True)
				if curr_q_id != prev_q_id:
					prev_q_id = curr_q_id
					self.stop = True
					break
				# extracting the token_ids and creating a numpy array
				text = self.get_text(line)

				tokens_list = self.tokenize(text)

				if id_ not in batch_ids:

					batch_ids.append(id_)

					batch_data.append(tokens_list)

				prev_q_id = curr_q_id

				line = self.file_.readline()

				#print(self.is_query, id_)

			batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
			#padd data along axis 1
			batch_data = pad_sequence(batch_data,1).long()

			yield batch_ids, batch_data, batch_lengths







class WeakSupervisionEval:
	def __init__(self, ranking_results_path, id2text, batch_size, is_query, min_len=5):
		# open file
		self.batch_size = batch_size
		self.ranking_results = open(ranking_results_path, 'r')
		self.is_query = is_query
		self.min_len = min_len
		if isinstance(id2text, FileInterface):
			self.id2text = id2text
		else:
			self.id2text = FileInterface(id2text)
		self.stop = False

	def reset(self):
		self.ranking_results.seek(0)
		return self


	def get_id(self, line, is_query):
		spl = line.split(' ')
		if is_query:
			return str(spl[0])
		else:
			return str(spl[2])

	def get_text(self, id_):
		tokenized_ids = self.id2text.get_tokenized_element(id_)
		if len(tokenized_ids) < self.min_len:
			tokenized_ids = np.pad(tokenized_ids, (0, self.min_len - len(tokenized_ids)))
		return tokenized_ids

	def batch_generator(self):

		line = self.ranking_results.readline()

		prev_q_id = self.get_id(line, is_query=True)
		curr_q_id = prev_q_id
		self.stop = False

		while line and not self.stop:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while (line and ( len(batch_ids) <= self.batch_size) ):

				id_ = self.get_id(line, self.is_query)
				curr_q_id = self.get_id(line, is_query=True)
				if curr_q_id != prev_q_id:
					prev_q_id = curr_q_id
					self.stop = True
					break
				# extracting the token_ids and creating a numpy array
				tokens_list = torch.IntTensor(self.get_text(id_))

				if id_ not in batch_ids:

					batch_ids.append(id_)

					batch_data.append(tokens_list)

				prev_q_id = curr_q_id

				line = self.ranking_results.readline()
			batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
			#padd data along axis 1
			batch_data = pad_sequence(batch_data,1).long()

			yield batch_ids, batch_data, batch_lengths

class WeakSupervisonTrain(data.Dataset):
	def __init__(self, weak_results_filename, documents_fi, queries_path, top_k_per_query=-1, sampler = 'random', target='binary'):


		# "open" triplets file
		self.weak_results_file = FileInterface(weak_results_filename)

		# "open" documents file
		self.documents = documents_fi
		# "open" queries file
		self.queries = FileInterface(queries_path)

		self.top_k_per_query = top_k_per_query


		if sampler == 'random':
			self.sample_function = self.random_sampler

		if target == 'binary':
			self.target_function = self.binary_target

		self.sampler = sampler
		self.target = target


	def __len__(self):
		return len(self.weak_results_file)


	def __getitem__(self, index):
		q_id, query_results = self.weak_results_file.read_all_results_of_query_index(index, self.top_k_per_query)

		d1_id, d2_id, target = self.get_sample_from_query_scores(query_results)

		# after tokenizing / preprocessing, some queries/documents have empty content.
		# If any of these are among 3 the selected ones, then we do not create this triplet sample
		# In that case, we return None, as soon as possible, so that other file reading operations can be avoided

		# retrieve tokenized content, given id
		query = self.queries.get_tokenized_element(q_id)
		if query is None:
			return None
		doc1 = self.documents.get_tokenized_element(d1_id)
		if doc1 is None:
			return None
		doc2 = self.documents.get_tokenized_element(d2_id)
		if doc2 is None:
			return None
		# shuffle order
		if np.random.random() > 0.5:
			return [query, doc1, doc2], 1
		else:
			return [query, doc2, doc1], -1


	def random_sampler(self, scores_list):
		return random.sample(population = scores_list, k=2)

	def binary_target(self, result1, result2):
		# 1 if result1 is better and -1 if result2 is better
		target = 1 if result1[1] > result2[1] else -1
		return  target

	def get_sample_from_query_scores(self, scores_list):

		result1, result2 = self.sample_function(scores_list)

		target = self.target_function(result1, result2)

		return result1[0], result2[0], target



def get_data_loaders_msmarco(cfg):

	cfg.msmarco_triplets_train = add_before_ending(cfg.msmarco_triplets_train,  '.debug' if cfg.debug else '')
	dataloaders = {}
	dataloaders['train'] = DataLoader(MSMarcoTrain(cfg.msmarco_triplets_train, cfg.msmarco_docs_train, cfg.msmarco_query_train),
	batch_size=cfg.batch_size, collate_fn=collate_fn_padd, shuffle=True, num_workers = cfg.num_workers)

	query_batch_generator = MSMarcoSequential(cfg.msmarco_query_val, cfg.batch_size)
	docs_batch_generator = MSMarcoSequential(cfg.msmarco_docs_val, cfg.batch_size)

	dataloaders['val'] = [query_batch_generator, docs_batch_generator]

	return dataloaders

def get_data_loaders_robust(cfg):
		
	docs_fi = FileInterface(cfg.robust_docs)
	cfg.robust_ranking_results_train = add_before_ending(cfg.robust_ranking_results_train,  '.debug' if cfg.debug else '')
	dataloaders = {}
	dataset = WeakSupervisonTrain(cfg.robust_ranking_results_train, docs_fi, cfg.robust_query_train, sampler = cfg.sampler, target=cfg.target)
	dataloaders['train'] = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_padd, shuffle=True, num_workers = cfg.num_workers)
	
	query_batch_generator = WeakSupervisionEval(cfg.robust_ranking_results_test, cfg.robust_query_test, cfg.batch_size, is_query=True)
	docs_batch_generator = WeakSupervisionEval(cfg.robust_ranking_results_test, docs_fi, cfg.batch_size, is_query=False)

	dataloaders['val'] = [query_batch_generator, docs_batch_generator]
	return dataloaders


class MSMarcoLM(data.Dataset):

	def __init__(self, data_path, documents_path, queries_path):


		self.data = open(data_path, 'r').readlines()

		# "open" documents file
		self.documents = FileInterface(documents_path)
		# "open" queries file
		self.queries = FileInterface(queries_path)


	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		q_id, _, d1_id, _  = self.data[index].split('\t')
		query = self.queries.get_tokenized_element(q_id)
		doc = self.documents.get_tokenized_element(d1_id)
		inp = list(query[1:]) + list(doc[1:])
		return torch.LongTensor(inp)
