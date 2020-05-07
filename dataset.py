import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path
from fake_data import *
from random import randint
from file_interface import FileInterface
from torch.utils.data import DataLoader, IterableDataset
from os import path
from utils import collate_fn_padd_triples, add_before_ending, collate_fn_padd_single
import math
import pickle
import random
from tokenizer import Tokenizer

class StrongData(IterableDataset):
	def __init__(self, strong_results_filename, documents_fi, queries_path, target):

		# "open" triplets file
		self.strong_results_file = FileInterface(strong_results_filename)

		# "open" documents file
		self.documents = documents_fi

		# create a list of docment ids
		self.doc_ids_list = list(self.documents.seek_dict)

		# "open" queries file
		self.queries = FileInterface(queries_path)


		if target == 'binary':
			self.target_function = self.binary_target
		elif target == 'rank_prob':
			self.target_function = self.probability_difference_target
		else:
			raise ValueError("Param 'target' of WeakSupervisonTrain, was not among {'binary', 'rank_prob'}, but :" + str( sampler))



	def __len__(self):
		return len(self.strong_results_file)

	def binary_target(self, result1, result2):
		# 1 if result1 is better and -1 if result2 is better
		target = 1 if result1[1] > result2[1] else -1
		return  target
		# implementation of the rank_prob model's target from paper : Neural Ranking Models with Weak Supervision (https://arxiv.org/abs/1704.08803)
	def probability_difference_target(self, result1, result2):
		target = result1[1] / (result1[1] + result2[1])
		return target

	# sampling candidates functions
	# sample a negative candidate from the collection
	def sample_negative_document_result(self, exclude_doc_ids_set):
		# get a random index from documents' list
		random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
		# get the corresponding document id
		random_doc_id = self.doc_ids_list[random_doc_index]
		# retrieve content of the random document
		document_content = self.documents.get_tokenized_element(random_doc_id)
		# make sure that the random document's id is not in the exclude list and its content is not empty
		while random_doc_id in exclude_doc_ids_set and document_content is not None:
		# get a random index from documents' list
			random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
		# get the corresponding document id
			random_doc_id = self.doc_ids_list[random_doc_index]
			# retrieve content of the random document
			document_content = self.documents.get_tokenized_element(random_doc_id)
		return (random_doc_id, 0, document_content)

	def __iter__(self):
		for index in range(len(self)):
			q_id, query_results = self.weak_results_file.read_all_results_of_query_index(index, top_k_per_query = -1)
			rel_docs, non_rel_docs = list(), list()

			for el in query_results:
				if el[1] > 0:
					rel_docs.append(el)
				else:
					non_rel_docs.append(el)

			rel_docs_set = {doc_id for doc_id, score in rel_docs}	

			for d1_id, score_1 in rel_docs:
				if np.random.random() > 0.5:
					if len(non_rel_docs) > 0:
						d2_id, d2_score = non_rel_docs.pop()
						content = self.documents.get_tokenized_element(d2_id)
						doc2 = (d2_id, d2_score, content )
						
					else:
						doc2 = self.sample_negative_document_result(rel_docs_set)
				else:
					doc2 = self.sample_negative_document_result(rel_docs_set)
	
				# after tokenizing / preprocessing, some queries/documents have empty content.
				# If any of these are among 3 the selected ones, then we do not create this triplet sample
				# In that case, we return None, as soon as possible, so that other file reading operations can be avoided

				# retrieve tokenized content, given id
				query = self.queries.get_tokenized_element(q_id)
				doc1 = self.documents.get_tokenized_element(d1_id)
	
				if doc1 is None:
					yield None
				if doc2[2] is None:
					yield None
					
				# randomly swap positions
				if np.random.random() > 0.5:
					temp = doc1
					doc1 = doc2
					doc2 = temp
				
				target = self.target_function(doc1, doc2)
				yield [query, doc1[2], doc2[2]], target

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


class MSMarcoSequential(IterableDataset):
	def __init__(self, fname):

		# open file
		self.file_ = open(fname, 'r')

	def __iter__(self):
		self.file_.seek(0)
		for line in self.file_:
			# getting position of '\t' that separates the doc_id and the begining of the token ids
			delim_pos = line.find('\t')
			# extracting the id
			id_ = line[:delim_pos]
			# extracting the token_ids and creating a numpy array
			tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')

			yield [id_, tokens_list]


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


class WeakSupervision(IterableDataset):
	def __init__(self, weak_results_filename, documents_fi, queries_path, top_k_per_query=-1, sampler = 'uniform', target='binary',
			samples_per_query = -1, single_sample = False, shuffle = True, min_results = 2, strong_negatives = True):

		# "open" triplets file
		self.weak_results_file = FileInterface(weak_results_filename)

		# "open" documents file
		self.documents = documents_fi

		# create a list of docment ids
		self.doc_ids_list = list(self.documents.seek_dict)

		# "open" queries file
		self.queries = FileInterface(queries_path)

		self.top_k_per_query = top_k_per_query
		# defines the full(~1000) combinations to be calculated for a number of (samples_per_query) queries
		self.samples_per_query = samples_per_query

		# if True, then we create exactly one positive sample for each query
		self.single_sample = single_sample
		# if strong_negatives == True then then reassuring that negative samples are not among the (weakly) relevant ones
		self.strong_negatives = strong_negatives

		self.shuffle = shuffle

		self.min_results = min_results

		if target == 'binary':
			self.target_function = self.binary_target
		elif target == 'rank_prob':
			self.target_function = self.probability_difference_target
		else:
			raise ValueError("Param 'target' of WeakSupervisonTrain, was not among {'binary', 'rank_prob'}, but :" + str( sampler))

		# setting a maximum of 2000 candidates to sample from, if not specified differently from top_k_per_query
		self.max_candidates = top_k_per_query if top_k_per_query !=-1 else 2000

		if sampler == 'top_n':
			self.sampler_function = self.sample_top_n
		elif sampler == 'uniform':
			self.sampler_function = self.sample_uniform
		elif sampler == 'zipf':
			# initialize common calculations
			self.sample_weights = np.asarray([1/(i+1) for i in range(self.max_candidates)])
			self.sampler_function = self.sample_zipf
		else:
			raise ValueError("Param 'sampler' of WeakSupervisonTrain, was not among {'top_n', 'uniform', 'zipf'}, but :" + str( sampler))
		# having a calculated list of indices, that will be used while sampling
		self.candidate_indices = [i for i in range(self.max_candidates)]

		# this will be used later in case suffle is True
		self.query_indices = [i for i in range(len(weak_results_file))]


	def generate_triplet(self, query, candidates):
		# shuffle order of candidates
		random.shuffle(candidates)
		# get target
		target = self.target_function(candidates)
		# get document content from tupples
		doc1 = candidates[0][2]
		doc2 = candidates[1][2]

		return [query, doc1, doc2], target


	def __iter__(self):

		if self.suffle:
			random.shuffle(self.query_indices)

		for q_index in self.query_indices:
			q_id, query_results = self.weak_results_file.read_all_results_of_query_index(q_index, self.top_k_per_query)

			# if the content of the query is empty, then skip this query
			query = self.queries.get_tokenized_element(q_id)
			if query is None:
				continue

			# make sure that there are not any empty documents on the retrieved documents list (query_results)
			# since we are reading the documents we are also saving their contents in memory as an extra item in the final tupples
			non_empty_query_results_with_content = []
			for doc_id, score in query_results:
				document_content = self.documents.get_tokenized_element(doc_id)
				if document_content is not None:
					# updating list with non empy_documents, and also adding the content of the document ot the tupple
					non_empty_query_results_with_content.append((doc_id, score, document_content))
			query_results = non_empty_query_results_with_content

			# skip queries that do not have the necessary nuber of results 
			if len(query_results) < self.min_results:
				continue

			# reassuring that negative results are not among the weak scorer results, by creting a set with all relevant ids
			if self.strong_negatives:
				relevant_doc_ids_set = {doc_id for doc_id , _, _ in query_results }

			#  if we are generating exactly one relevant sample for each query (and one negative)
			if self.single_sample:

				# sample candidates
				candidates = self.sampler_function(scores_list = query_results, n = 2)

				# yield triplet of relevants
				yield self.generate_triplet(query, candidates)

				# get the first of the candidates in order to be matched with a random negative document
				result1 = candidates[0]

				# add the relevant document id to the excluding list if we haven't already
				if self.strong_negatives == False:
					rel_doc_id = result1[0]
					relevant_doc_ids_set = {rel_doc_id}

				negative_result = self.sample_negative_document_result(exclude_doc_ids_set = relevant_doc_ids_set)

				yield self.generate_triplet(query, [result1, negative_result])

			#  if we are generating all combinations from samples_per_query candidates with all the candidates samples
			# (plus 1 negative sample for each of the afforementioned samples)
			else:

				if self.samples_per_query == -1:
					candidate_indices = [i for i in range( len(query_results) )]
				else:
					candidate_indices = self.sampler_function(scores_list = query_results, n = self.samples_per_query, return_indices = True)
					candidate_indices.sort()

				# generating a sample for each combination of i_th candidate with j_th candidate, without duplicates 
				for i in candidate_indices:
					for j in range(len(query_results)):
						# making sure that we do not have any duplicates
						if (j not in candidate_indices) or (j > i):

							# yield triplet of relevants
							candidate1 = query_results[i]
							candidate2 = query_results[j]
							yield self.generate_triplet(query, [candidate1, candidate2])

							# yield triplet of irrelevants
							# add the relevant document id to the excluding list if we haven't already
							if self.strong_negatives == False:
								rel_doc_id = candidate1[0]
								relevant_doc_ids_set = {rel_doc_id}

							negative_result = self.sample_negative_document_result(exclude_doc_ids_set = relevant_doc_ids_set)

							yield self.generate_triplet(query, [candidate1, negative_result])


# target value calculation functions
	# binary targets -1/1 defining which is the more relevant candidate out of the two candidates
	def binary_target(self, result1, result2):
		# 1 if result1 is better and -1 if result2 is better
		target = 1 if result1[1] > result2[1] else -1
		return  target

	# implementation of the rank_prob model's target from paper : Neural Ranking Models with Weak Supervision (https://arxiv.org/abs/1704.08803)
	def probability_difference_target(self, result1, result2):
		target = result1[1] / (result1[1] + result2[1])
		return target

# sampling candidates functions
	# sample a negative candidate from the collection
	def sample_negative_document_result(self, exclude_doc_ids_set):
		# get a random index from documents' list
		random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
		# get the corresponding document id
		random_doc_id = self.doc_ids_list[random_doc_index]
		# retrieve content of the random document
		document_content = self.documents.get_tokenized_element(random_doc_id)

		# make sure that the random document's id is not in the exclude list and its content is not empty
		while random_doc_id in exclude_doc_ids_set and document_content is not None:
		# get a random index from documents' list
			random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
		# get the corresponding document id
			random_doc_id = self.doc_ids_list[random_doc_index]
			# retrieve content of the random document
			document_content = self.documents.get_tokenized_element(random_doc_id)

		return (random_doc_id, 0, document_content)

	# sampling out of relevant documents functions :

	def sample_uniform(self, scores_list, n, return_indices = False):
		length = len(scores_list)
		indices = self.candidate_indices[:length]
		sampled_indices = np.random.choice(indices, size=n, replace=False)
		if return_indices:
			return sampled_indices
		return [scores_list[i] for i in sampled_indices]

	def sample_zipf(self, scores_list, n, return_indices = False):
		length = len(scores_list)
		indices = self.candidate_indices[:length]
		# normalize sampling probabilities depending on the number of candidates
		p = self.sample_weights[:length] / sum(self.sample_weights[:length])
		sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
		if return_indices:
			return sampled_indices
		return [scores_list[i] for i in sampled_indices]

	def sample_top_n(self, scores_list, n, return_indices = False):
		if return_indices:
			return [i for i in range(n)]
		return scores_list[:n]

	#  def sample linear



class WeakSupervisonTrain(WeakSupervision, data.Dataset):
	def __init__(self, weak_results_filename, documents_fi, queries_path, top_k_per_query=-1, sampler = 'uniform', target='binary'):


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

		return [query, doc1, doc2], target

	def sample_with_negative(self, scores_list):

		# sample a relevant document.
		# In this case, the random relevant document is always being sampled uniformly!
		relevant_sample_result = self.sample_uniform(scores_list = scores_list, n = 1)[0]

		relevant_doc_id = relevant_sample_result[0]

		# sample a random doument from all documents
		random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)

		random_doc_id = self.doc_ids_list[random_doc_index]

		# make sure the two document ids are different
		while relevant_doc_id == random_doc_id:
			random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
			random_doc_id = self.doc_ids_list[random_doc_index]

		random_doc_result = (random_doc_id, 0)

		return relevant_sample_result, random_doc_result


	def get_sample_from_query_scores(self, scores_list):

		if np.random.random() > 0.5:
			# sample from relevant ones
			result1, result2 = self.sampler_function(scores_list = scores_list, n = 2)
		else:
			# sample one relevant and one negative
			result1, result2 = self.sample_with_negative(scores_list)

		# randomly swap positions
		if np.random.random() > 0.5:
			temp = result1
			result1 = result2
			result2 = temp

		target = self.target_function(result1, result2)

		return result1[0], result2[0], target








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

def split_dataset(train_val_ratio, dataset):
	lengths = [math.floor(len(dataset)*train_val_ratio), math.ceil(len(dataset)*(1-train_val_ratio))]
	# split dataset into train and test
	train_dataset, validation_dataset = torch.utils.data.dataset.random_split(dataset, lengths)
	return train_dataset, validation_dataset

def get_data_loaders_msmarco(cfg):

	cfg.msmarco_triplets_train = add_before_ending(cfg.msmarco_triplets_train,  '.debug' if cfg.debug else '')
	dataloaders = {}
	dataset = MSMarcoTrain(cfg.msmarco_triplets_train, cfg.msmarco_docs_train, cfg.msmarco_query_train)
	
	train_dataset, validation_dataset = split_dataset(train_val_ratio=0.9, dataset=dataset)
	
	dataloaders['train'] = DataLoader(train_dataseti,
	                                  batch_size=cfg.batch_size, collate_fn=collate_fn_padd_triples, shuffle=True, num_workers = cfg.num_workers)
	dataloaders['val'] = DataLoader(validation_dataset,
	                                  batch_size=cfg.batch_size, collate_fn=collate_fn_padd_triples, shuffle=False, num_workers = cfg.num_workers)


	sequential_num_workers = 1 if cfg.num_workers > 0 else 0


	query_batch_generator = DataLoader(MSMarcoSequential(cfg.msmarco_query_val), batch_size=cfg.batch_size, collate_fn=collate_fn_padd_single, num_workers = sequential_num_workers)
	docs_batch_generator = DataLoader(MSMarcoSequential(cfg.msmarco_docs_val), batch_size=cfg.batch_size, collate_fn=collate_fn_padd_single, num_workers = sequential_num_workers)

	dataloaders['test'] = [query_batch_generator, docs_batch_generator]

	return dataloaders

def get_data_loaders_robust(cfg):
	docs_fi = FileInterface(cfg.robust_docs)
	cfg.robust_ranking_results_train = add_before_ending(cfg.robust_ranking_results_train,  '.debug' if cfg.debug else '')
	dataloaders = {}
	dataset = WeakSupervisonTrain(cfg.robust_ranking_results_train, docs_fi, cfg.robust_query_train, sampler = cfg.sampler, target=cfg.target)
	# calculate train and validation size according to train_val_ratio
	train_dataset, validation_dataset = split_dataset(train_val_ratio=0.9, dataset=dataset)
	dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_padd_triples, shuffle=True, num_workers = cfg.num_workers)
	dataloaders['val'] = DataLoader(validation_dataset, batch_size=cfg.batch_size, collate_fn=collate_fn_padd_triples, shuffle=False, num_workers = cfg.num_workers)

	sequential_num_workers = 1 if cfg.num_workers > 0 else 0

	query_batch_generator = WeakSupervisionEval(cfg.robust_ranking_results_test, cfg.robust_query_test, cfg.batch_size, is_query=True, num_workers = sequential_num_workers)
	docs_batch_generator = WeakSupervisionEval(cfg.robust_ranking_results_test, docs_fi, cfg.batch_size, is_query=False, num_workers = sequential_num_workers)
	dataloaders['test'] = [query_batch_generator, docs_batch_generator]

	return dataloaders
