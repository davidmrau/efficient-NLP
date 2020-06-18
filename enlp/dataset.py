
import random
import torch
from enlp.tokenizer import Tokenizer
from torch.utils import data
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from enlp.file_interface import FileInterface
from enlp.utils import collate_fn_padd_triples, offset_dict_len, split_by_len, split_dataset

from torch.nn.utils.rnn import pad_sequence


class Sequential(IterableDataset):
	def __init__(self, fname, tokenize=False, min_len=5):

		# open file
		self.file_ = open(fname, 'r')
		self.tokenize = tokenize
		self.min_len = min_len
		self.tokenizer = Tokenizer(tokenizer = 'glove', max_len = 150, stopwords='lucene', remove_unk = True, unk_words_filename=None)

	def __iter__(self):
			for line in self.file_:
				# getting position of '\t' that separates the doc_id and the begining of the token ids
				delim_pos = line.find('\t')
				# extracting the id
				id_ = line[:delim_pos]
				# extracting the token_ids and creating a numpy array
				if self.tokenize:
					tokens_list = self.tokenizer.encode(line[delim_pos+1:])
				else:
					tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')

				if len(tokens_list) < self.min_len:
					tokens_list = np.pad(tokens_list, (0, self.min_len - len(tokens_list)))

				yield [id_, tokens_list]

class StrongData(IterableDataset):
	def __init__(self, strong_results, documents_fi, queries, target, indices=None):

		# "open" triplets file
		if isinstance(strong_results, FileInterface):
			self.strong_results_file = strong_results
		else:
			self.strong_results_file = FileInterface(strong_results)

		# "open" documents file
		self.documents = documents_fi
		if indices == None:
			self.indices = list(range(len(self)))
		else:
			self.indices = indices

		# create a list of docment ids
		self.doc_ids_list = list(self.documents.seek_dict)

		# "open" queries file

		if isinstance(queries, FileInterface):
			self.queries = queries
		else:
			self.queries = FileInterface(queries)

		if target == 'binary':
			self.target_function = self.binary_target
		elif target == 'rank_prob':
			self.target_function = self.probability_difference_target
		else:
			raise ValueError("Param 'target' of StrongData, was not among {'binary', 'rank_prob'}, but :" + str( target))


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
		while random_doc_id in exclude_doc_ids_set or document_content is None:
		# get a random index from documents' list
			random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
		# get the corresponding document id
			random_doc_id = self.doc_ids_list[random_doc_index]
			# retrieve content of the random document
			document_content = self.documents.get_tokenized_element(random_doc_id)
		return (random_doc_id, 0, document_content)

	def __iter__(self):
		for index in self.indices:
			q_id, query_results = self.strong_results_file.read_all_results_of_query_index(index, top_k_per_query = -1)
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
						content_2 = self.documents.get_tokenized_element(d2_id)
						result_2 = (d2_id, d2_score, content_2)

					else:
						result_2 = self.sample_negative_document_result(rel_docs_set)
				else:
					result_2 = self.sample_negative_document_result(rel_docs_set)

				# after tokenizing / preprocessing, some queries/documents have empty content.
				# If any of these are among 3 the selected ones, then we do not create this triplet sample
				# In that case, we return None, as soon as possible, so that other file reading operations can be avoided

				# retrieve tokenized content, given id
				query = self.queries.get_tokenized_element(q_id)
				content_1 = self.documents.get_tokenized_element(d1_id)
				result_1 = (d1_id, score_1, content_1)
				if result_1[2] is None or result_2[2] is None:
					yield None
					continue
				# randomly swap positions
				if np.random.random() > 0.5:
					temp = result_1
					result_1 = result_2
					result_2 = temp

				target = self.target_function(result_1, result_2)
				yield [query, result_1[2], result_2[2]], target

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

class RankingResultsTest_leg:

	def __init__(self, ranking_results, id2text, batch_size, is_query, min_len=5, indices=None):
		# open file
		self.batch_size = batch_size

		if isinstance(ranking_results, FileInterface):
			self.ranking_results = ranking_results
		else:
			self.ranking_results = open(ranking_results, 'r')
		self.is_query = is_query
		self.min_len = min_len
		self.indices = indices if indices else []
		if isinstance(id2text, FileInterface):
			self.id2text = id2text
		else:
			self.id2text = FileInterface(id2text)
		self.stop = False
		self.index = -1

	def reset(self):
		self.ranking_results.seek(0)
		self.line = None
		return self


	def get_id(self, line, is_query):
		spl = line.split(' ')
		if is_query:
			return str(spl[0].strip())
		else:
			return str(spl[2].strip())

	def get_text(self, id_):
		tokenized_ids = self.id2text.get_tokenized_element(id_)
		if tokenized_ids is None:
			return None

		if len(tokenized_ids) < self.min_len:
			tokenized_ids = np.pad(tokenized_ids, (0, self.min_len - len(tokenized_ids)))
		return tokenized_ids

	def batch_generator(self):

		if self.line is None:
			self.line = self.ranking_results.readline()

		prev_q_id = self.get_id(self.line, is_query=True)
		self.stop = False

		while self.line and not self.stop:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while (self.line and ( len(batch_ids) < self.batch_size) ):

				id_ = self.get_id(self.line, self.is_query)
				curr_q_id = self.get_id(self.line, is_query=True)
				if curr_q_id != prev_q_id:
					prev_q_id = curr_q_id
					self.stop = True
					self.index += 1
					break
				# extracting the token_ids and creating a numpy array




				if id_ not in batch_ids and self.index in self.indices:
					tokens_list = self.get_text(id_)
					if tokens_list is not None:
						batch_ids.append(id_)
						batch_data.append(torch.IntTensor(tokens_list))

				prev_q_id = curr_q_id

				self.line = self.ranking_results.readline()

			batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
			if len(batch_data) < 1:
				return
			#padd data along axis 1
			batch_data = pad_sequence(batch_data,1).long()

			yield batch_ids, batch_data, batch_lengths


class RankingResultsTest:

	def __init__(self, ranking_results, id2query, id2doc, batch_size, min_len=5, indices=None):
		# open file
		self.batch_size = batch_size

		if isinstance(ranking_results, FileInterface):
			self.ranking_results = ranking_results
		else:
			self.ranking_results = open(ranking_results, 'r')
		self.min_len = min_len
		self.indices = indices if indices else []
		if isinstance(id2query, FileInterface):
			self.id2query = id2query
		else:
			self.id2query = FileInterface(id2query)

		if isinstance(id2doc, FileInterface):
			self.id2doc = id2doc
		else:
			self.id2doc = FileInterface(id2doc)

		self.stop = False
		self.index = -1

	def reset(self):
		self.ranking_results.seek(0)
		self.line = None
		return self


	def get_id(self, line):
		spl = line.split(' ')
		q_id = str(spl[0].strip())
		d_id = str(spl[2].strip())
		return q_id, d_id

	def get_tokenized(self, id_, id2tokens):
		tokenized_ids = id2tokens.get_tokenized_element(id_)
		if tokenized_ids is None:
			return None

		if len(tokenized_ids) < self.min_len:
			tokenized_ids = np.pad(tokenized_ids, (0, self.min_len - len(tokenized_ids)))
		return tokenized_ids

	def batch_generator(self):

		if self.line is None:
			self.line = self.ranking_results.readline()

		prev_q_id, _ = self.get_id(self.line)
		self.stop = False

		while self.line and not self.stop:
			# read a number of lines equal to batch_size
			d_batch_ids = []
			d_batch_data = []
			while (self.line and ( len(d_batch_ids) < self.batch_size) ):

				curr_q_id, doc_id = self.get_id(self.line)
				if curr_q_id != prev_q_id:
					prev_q_id = curr_q_id
					self.stop = True
					self.index += 1
					break
				# extracting the token_ids and creating a numpy array



				# if index of queries to load is provided check if query index is indices
				if self.index in self.indices:
					doc = self.get_tokenized(doc_id, self.id2doc)
					if doc is not None:
						d_batch_ids.append(doc_id)
						d_batch_data.append(torch.IntTensor(doc))

				prev_q_id = curr_q_id


				self.line = self.ranking_results.readline()

			query = self.get_tokenized(curr_q_id, self.id2query)	
			q_data = [torch.IntTensor(query)]
			q_id = [curr_q_id]
			d_batch_lengths = torch.FloatTensor([len(d) for d in d_batch_data])
			q_length = torch.FloatTensor([len(q) for q in q_data])
	
			if len(d_batch_data) < 1:
				return
			#padd data along axis 1
			d_batch_data = pad_sequence(d_batch_data,1).long()
			q_data = pad_sequence(q_data,1).long()

			yield q_id, q_data, q_length, d_batch_ids, d_batch_data, d_batch_lengths


class WeakSupervision(IterableDataset):
	def __init__(self, weak_results_fi, documents_fi, queries_fi, top_k_per_query=-1, sampler = 'uniform', target='binary',
			samples_per_query = -1, single_sample = False, shuffle = True, min_results = 2, strong_negatives = True, indices_to_use = None,
			sample_j = False):

		# "open" triplets file
		self.weak_results_file = weak_results_fi

		# "open" documents file
		self.documents = documents_fi

		# create a list of docment ids
		self.doc_ids_list = list(self.documents.seek_dict)

		# "open" queries file
		self.queries = queries_fi

		self.top_k_per_query = top_k_per_query
		# defines the full(~1000) combinations to be calculated for a number of (samples_per_query) queries
		self.samples_per_query = samples_per_query

		# if True, then we create exactly one positive sample for each query
		self.single_sample = single_sample
		# if strong_negatives == True then then reassuring that negative samples are not among the (weakly) relevant ones
		self.strong_negatives = strong_negatives

		self.shuffle = shuffle
		# if sample_j is True, then we sample samples_per_query samples for creating the cpmbinations. sample different ones for each (i)
		self.sample_j = sample_j

		self.min_results = min_results

		if target == 'binary':
			self.target_function = self.binary_target
		elif target == 'rank_prob':
			self.target_function = self.probability_difference_target
		else:
			raise ValueError("Param 'target' of WeakSupervision, was not among {'binary', 'rank_prob'}, but :" + str( target))

		# setting a maximum of 2000 candidates to sample from, if not specified differently from top_k_per_query
		self.max_candidates = top_k_per_query if top_k_per_query !=-1 else 2000

		if sampler == 'top_n':
			self.sampler_function = self.sample_top_n
		elif sampler == 'uniform':
			self.sampler_function = self.sample_uniform
		elif sampler == 'linear':
			self.sample_weights =  np.linspace(1,0,self.max_candidates)
			self.sampler_function = self.sample_linear
		elif sampler == 'zipf':
			# initialize common calculations
			self.sample_weights = np.asarray([1/(i+1) for i in range(self.max_candidates)])
			self.sampler_function = self.sample_zipf
		else:
			raise ValueError("Param 'sampler' of WeakSupervision, was not among {'top_n', 'uniform', 'zipf', 'linear'}, but :" + str( sampler))
		# having a calculated list of indices, that will be used while sampling
		self.candidate_indices = list(range(self.max_candidates))

		# this will be used later in case suffle is True
		if indices_to_use is None:
			self.query_indices = list(range(len(self.weak_results_file)))
		else:
			self.query_indices = indices_to_use

	def __len__(self):
		raise NotImplementedError()

	def generate_triplet(self, query, candidates):
		# shuffle order of candidates
		random.shuffle(candidates)
		# get target
		target = self.target_function(candidates[0], candidates[1])
		# get document content from tupples
		doc1 = candidates[0][2]
		doc2 = candidates[1][2]
		return [query, doc1, doc2], target


	def __iter__(self):

		if self.shuffle:
			random.shuffle(self.query_indices)

		for q_index in self.query_indices:
			q_id, query_results = self.weak_results_file.read_all_results_of_query_index(q_index, self.top_k_per_query)

			# if the content of the query is empty, then skip this query
			query = self.queries.get_tokenized_element(q_id)
			if query is None:
				continue

			# skip queries that do not have the necessary nuber of results
			if len(query_results) < self.min_results:
				continue

			# reassuring that negative results are not among the weak scorer results, by creting a set with all relevant ids
			if self.strong_negatives:
				relevant_doc_ids_set = {doc_id for doc_id , _ in query_results }

			#  if we are generating exactly one relevant sample for each query (and one negative)
			if self.single_sample:

				# sample candidates
				candidate_indices = self.sampler_function(scores_list = query_results, n = 2, return_indices = True)

				doc1_id, score1 = query_results[candidate_indices[0]]
				doc2_id, score2 = query_results[candidate_indices[1]]

				doc1 = self.documents.get_tokenized_element(doc1_id)
				doc2 = self.documents.get_tokenized_element(doc2_id)

				candidates = [(doc1_id, score1, doc1), (doc2_id, score2, doc2)]

				if (doc1 is not None) and (doc2 is not None):
					# yield triplet of relevants
					yield self.generate_triplet(query, candidates)

				else:
					continue

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

				# make sure that there are not any empty documents on the retrieved documents list (query_results)
				# since we are reading the documents we are also saving their contents in memory as an extra item in the final tupples
				non_empty_query_results_with_content = []
				for doc_id, score in query_results:
					document_content = self.documents.get_tokenized_element(doc_id)
					if document_content is not None:
						# updating list with non empy_documents, and also adding the content of the document ot the tupple
						non_empty_query_results_with_content.append((doc_id, score, document_content))
				query_results = non_empty_query_results_with_content


				# inb case we will end up using all the candidaes to create combinations
				if self.samples_per_query == -1 or len(query_results) <= self.samples_per_query :
					candidate_indices = [i for i in range( len(query_results) )]
				else:
					candidate_indices = self.sampler_function(scores_list = query_results, n = self.samples_per_query, return_indices = True)
					candidate_indices.sort()

				# generating a sample for each combination of i_th candidate with j_th candidate, without duplicates
				for i in candidate_indices:
					# if we do not request sampling, or there are not enough results to sample from, then we use all of them
					if (self.sample_j == False) or (self.samples_per_query == -1) or (len(query_results) <= self.samples_per_query):
						j_indices = list(range(len(query_results)))
					# otherwise we are able and requested to sample for the nested loop of combinations (j), so we do sample
					else:
						j_indices = self.sampler_function(scores_list = query_results, n = self.samples_per_query, return_indices = True)

					for j in j_indices:
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
		while random_doc_id in exclude_doc_ids_set or document_content is None:
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

	def sample_linear(self, scores_list, n, return_indices = False):
		length = len(scores_list)
		indices = self.candidate_indices[:length]
		# normalize sampling probabilities depending on the number of candidates
		p = self.sample_weights[:length] / sum(self.sample_weights[:length])
		sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
		if return_indices:
			return sampled_indices
		return [scores_list[i] for i in sampled_indices]







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

def get_data_loaders_msmarco(cfg):

	if cfg.debug:
		triples = cfg.msmarco_triplets_train_debug
	else:
		triples = cfg.msmarco_triplets_train
	print(triples)
	dataloaders = {}
	dataset = MSMarcoTrain(triples, cfg.msmarco_docs_train, cfg.msmarco_query_train)

	train_dataset, validation_dataset = split_dataset(train_val_ratio=0.9, dataset=dataset)

	dataloaders['train'] = DataLoader(train_dataset,
	                                  batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, shuffle=True, num_workers = cfg.num_workers)
	dataloaders['val'] = DataLoader(validation_dataset,
	                                  batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, shuffle=False, num_workers = cfg.num_workers)


	sequential_num_workers = 1 if cfg.num_workers > 0 else 0

	query_batch_generator = RankingResultsTest(cfg.msmarco_ranking_results_test, cfg.msmarco_query_test, cfg.batch_size_test, is_query=True)

	docs_batch_generator = RankingResultsTest(cfg.msmarco_ranking_results_test, cfg.msmarco_docs_test, cfg.batch_size_test, is_query=False)
	dataloaders['test'] = [query_batch_generator, docs_batch_generator]

	return dataloaders


def get_data_loaders_robust(cfg):
	if cfg.debug:
		ranking_results_train = cfg.robust_ranking_results_train_debug
	else:
		ranking_results_train = cfg.robust_ranking_results_train

	docs_fi = FileInterface(cfg.robust_docs)
	queries_fi = FileInterface(cfg.robust_query_train)
	weak_results_fi = FileInterface(ranking_results_train)
	dataloaders = {}

	dataset_len = offset_dict_len(ranking_results_train)

	indices_train, indices_val = split_by_len(dataset_len, ratio = 0.9)
	# dataset = WeakSupervision(cfg.robust_ranking_results_train, docs_fi, cfg.robust_query_train, sampler = cfg.sampler, target=cfg.target)

	train_dataset = WeakSupervision(weak_results_fi, docs_fi, queries_fi, sampler = cfg.sampler, target=cfg.target, single_sample=cfg.single_sample,
					 shuffle=True, indices_to_use = indices_train, samples_per_query = cfg.samples_per_query, sample_j = cfg.sample_j, min_results=cfg.weak_min_results)

	validation_dataset = WeakSupervision(weak_results_fi, docs_fi, queries_fi, sampler = 'uniform', target=cfg.target, single_sample = True,
					shuffle=False, indices_to_use = indices_val, samples_per_query = cfg.samples_per_query, min_results=cfg.weak_min_results)

	sequential_num_workers = 1 if cfg.num_workers > 0 else 0

	# calculate train and validation size according to train_val_ratio
	# train_dataset, validation_dataset = split_dataset(train_val_ratio=0.9, dataset=dataset)
	dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, num_workers = sequential_num_workers)
	dataloaders['val'] = DataLoader(validation_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples,  num_workers = sequential_num_workers)


	query_batch_generator = RankingResultsTest(cfg.robust_ranking_results_test, cfg.robust_query_test, cfg.batch_size_test, is_query=True)
	docs_batch_generator = RankingResultsTest(cfg.robust_ranking_results_test, docs_fi, cfg.batch_size_test, is_query=False)
	dataloaders['test'] = [query_batch_generator, docs_batch_generator]

	return dataloaders

def get_data_loaders_robust_strong(cfg, indices_train, indices_test, docs_fi, query_fi, ranking_results_fi):


	dataloaders = {}

	# calculate train and validation size according to train_val_ratio

	sequential_num_workers = 1 if cfg.num_workers > 0 else 0
	dataloaders['train'] = DataLoader(StrongData(ranking_results_fi, docs_fi, query_fi, indices=indices_train, target=cfg.target), batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, num_workers = sequential_num_workers)
	dataloaders['test'] = RankingResultsTest(cfg.robust_ranking_results_test, query_fi,  docs_fi, cfg.batch_size_test, indices=indices_test)
	return dataloaders
