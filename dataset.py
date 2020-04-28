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


class MSMarco(data.Dataset):

	def __init__(self, split, triplets_path, documents_path, queries_path, debug=False):

		self.split = split
		self.debug = debug

		triplets_path = add_before_ending(triplets_path, '.debug' if debug else '')
		# "open" triplets file
		self.triplets = FileInterface(triplets_path)


		# "open" documents file
		self.documents = FileInterface(documents_path)
		# "open" queries file
		self.queries = FileInterface(queries_path)
		# read qrels
		# self.qrels = read_qrels(f'{dataset_path}/qrels.{split}.tsv')

		self.max_doc_id = len(self.documents) - 1

	def __len__(self):
		if self.split == 'train':
			return len(self.triplets)

		elif self.split == 'dev':
			return len(self.qrels)


	def __getitem__(self, index):

		if self.split == 'train':

			q_id, d1_id, d2_id = self.triplets.get_triplet(index)

		elif self.split == 'dev':
			pass
			# q_id, d1_id = self.qrels[index]
			#
			# d2_id = randint(0, self.max_doc_id)
			# # making sure that the sampled d1_id is diferent from relevant d2_id
			# while int(d1_id) == d2_id:
			# 	d2_id = randint(0, self.max_doc_id)
			# # type cast the id into a string
			# d2_id = str(d2_id)
		else:
			raise ValueError(f'Unknown split: {split}')

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
		self.fname = fname
		self.file_ = None

	def reset(self):
		self.file_ = open(self.fname, 'r')

	def batch_generator(self):

		line = self.file_.readline()
		# line = self.file.readline()
		while line:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while(line and ( len(batch_ids) < self.batch_size) ):

				# getting position of first ' ' that separates the doc_id and the begining of the token ids
				delim_pos = line.find(' ')
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
	def __init__(self, fname, batch_size, word2index_path, embedding, is_query, min_len=5, max_len=150, stopwords = "lucene", remove_unk = True):

		# open file
		self.batch_size = batch_size
		self.fname = fname
		self.is_query = is_query
		self.min_len = min_len

		self.tokenizer = 	tokenizer = Tokenizer(tokenizer = embedding, max_len = max_len, stopwords=stopwords, remove_unk = remove_unk,
							word2index_path = word2index_path, unk_words_filename = None)

		self.file = None
		self.stop = False

	def reset(self):
		self.file_ = open(self.fname, 'r')
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


class WeakSupervisonTrain(data.Dataset):
	def __init__(self, weak_results_filename = "data/AOL_BM25.results", batch_size=3, top_k_per_query=3, sampler = 'random', target='binary'):

		# open weak supervision
		self.weak_results_file = open(weak_results_filename)

		self.batch_size = batch_size
		self.top_k_per_query = top_k_per_query


		if sampler == 'random':
			self.sample_function = self.random_sampler

		if target == 'binary':
			self.target_function = self.binary_target


		self.sampler = sampler
		self.target = target


		self.prev_q_id = None


	def random_sampler(self, scores_list):
		return random.sample(population = scores_list, k=2)

	def binary_target(self, result1, result2):
		# 1 if result1 is better and -1 if result2 is better
		target = 1 if result1[1] > result2[1] else -1
		return  target

	def get_sample_from_query_scores(self, q_id, scores_list):

		result1, result2 = self.sample_function(scores_list)

		target = self.target_function(result1, result2)

		return q_id, result1[0], result2[0], target


	def batch_generator(self):


		batch = []

		while True:

			line = self.weak_results_file.readline()
			# initialize with first query
			prev_q_id = line.split()[0]
			q_id = prev_q_id

			scores_list = []

			# if we have read the complete file
			while line:

				# for line in fp:
				split_line = line.split()

				q_id = split_line[0]
				doc_id = split_line[2]
				# rank = split_line[3]
				score = float(split_line[4])

				if q_id != prev_q_id:

					# if len(scores_list) == 0: Remember to make sure that it works if the results provided are less than the results requested

					batch.append( self.get_sample_from_query_scores(prev_q_id, scores_list) )
					scores_list = []

					if len(batch) == self.batch_size:
						yield batch
						batch = []

				# only using the keep_top_k relevant docs for each query
				if len(scores_list) != self.top_k_per_query:
					# add this relevant document with its score to the list of relevant documents for this query
					scores_list.append((doc_id, score))


				prev_q_id = q_id

				line = self.weak_results_file.readline()

			# continue reading the file from the beginning
			self.weak_results_file.seek(0)

			# using the last query on the file
			batch.append( self.get_sample_from_query_scores(prev_q_id, scores_list) )
			scores_list = []

			if len(batch) == self.batch_size:
				yield batch
				batch = []




def get_data_loaders(triplets_train_file, docs_file_train, query_file_train, query_file_val,
 	docs_file_val, batch_size, num_workers, debug=False):

	dataloaders = {}
	dataloaders['train'] = DataLoader(MSMarco('train', triplets_train_file, docs_file_train, query_file_train, debug=debug),
	batch_size=batch_size, collate_fn=collate_fn_padd, shuffle=True, num_workers = num_workers)

	query_batch_generator = MSMarcoSequential(query_file_val, batch_size)
	docs_batch_generator = MSMarcoSequential(docs_file_val, batch_size)

	dataloaders['val'] = [query_batch_generator, docs_batch_generator]

	#dataloaders['test'] =  DataLoader(MSMarco('eval'), batch_size=batch_size, collate_fn=collate_fn_padd)

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


