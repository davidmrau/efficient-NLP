
import random
import torch
from enlp.tokenizer import Tokenizer
from torch.utils import data
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from enlp.file_interface import FileInterface
from enlp.utils import collate_fn_padd_triples, offset_dict_len, split_by_len, split_dataset, padd_tensor, \
				collate_fn_bert_interaction, create_bert_inretaction_input
				

from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence


class Sequential(IterableDataset):
	def __init__(self, fname, tokenize=False, min_len=5, max_len=150, stopwords='lucene', tokenizer='glove', remove_unk=True):

		# open file
		self.file_ = open(fname, 'r')
		self.tokenize = tokenize
		self.min_len = min_len
		self.tokenizer = Tokenizer(tokenizer = tokenizer, max_len = max_len, stopwords=stopwords, remove_unk = remove_unk, unk_words_filename=None)

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

# currently StrongData is not used
# If it is to be used, the parameters: {max_query_len = 64, max_complete_length = 510, max_doc_len = None} need to be created and set according to :
# cfg params :
# msmarco:
#   max_query_len: 64
#   max_complete_length: 512
# robust04:
#   max_length: 1500

# class StrongData(IterableDataset):
# 	def __init__(self, strong_results, documents_fi, queries, target, indices=None, sample_random=True):

# 		# "open" triplets file
# 		if isinstance(strong_results, FileInterface):
# 			self.strong_results_file = strong_results
# 		else:
# 			self.strong_results_file = FileInterface(strong_results)

# 		# "open" documents file
# 		self.documents = documents_fi
# 		if indices == None:
# 			self.indices = list(range(len(self)))
# 		else:
# 			self.indices = indices

# 		# create a list of docment ids
# 		self.doc_ids_list = list(self.documents.seek_dict)

# 		# "open" queries file

# 		if isinstance(queries, FileInterface):
# 			self.queries = queries
# 		else:
# 			self.queries = FileInterface(queries)

# 		if target == 'binary':
# 			self.target_function = self.binary_target
# 		elif target == 'rank_prob':
# 			self.target_function = self.probability_difference_target
# 		else:
# 			raise ValueError("Param 'target' of StrongData, was not among {'binary', 'rank_prob'}, but :" + str( target))
# 		self.sample_random = sample_random

# 	def __len__(self):
# 		return len(self.strong_results_file)

# 	def binary_target(self, result1, result2):
# 		# 1 if result1 is better and -1 if result2 is better
# 		target = 1 if result1[1] > result2[1] else -1
# 		return  target
# 		# implementation of the rank_prob model's target from paper : Neural Ranking Models with Weak Supervision (https://arxiv.org/abs/1704.08803)
# 	def probability_difference_target(self, result1, result2):
# 		target = result1[1] / (result1[1] + result2[1])
# 		return target

# 	# sampling candidates functions
# 	# sample a negative candidate from the collection
# 	def sample_negative_document_result(self, exclude_doc_ids_set):
# 		# get a random index from documents' list
# 		random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
# 		# get the corresponding document id
# 		random_doc_id = self.doc_ids_list[random_doc_index]
# 		# retrieve content of the random document
# 		document_content = self.documents.get_tokenized_element(random_doc_id)
# 		# make sure that the random document's id is not in the exclude list and its content is not empty
# 		while random_doc_id in exclude_doc_ids_set or document_content is None:
# 		# get a random index from documents' list
# 			random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
# 		# get the corresponding document id
# 			random_doc_id = self.doc_ids_list[random_doc_index]
# 			# retrieve content of the random document
# 			document_content = self.documents.get_tokenized_element(random_doc_id)
# 		return (random_doc_id, 0, document_content)

# 	def __iter__(self):
# 		for index in self.indices:
# 			q_id, query_results = self.strong_results_file.read_all_results_of_query_index(index, top_k_per_query = -1)
# 			rel_docs, non_rel_docs = list(), list()

# 			for el in query_results:
# 				if el[1] > 0:
# 					rel_docs.append(el)
# 				else:
# 					non_rel_docs.append(el)

# 			rel_docs_set = {doc_id for doc_id, score in rel_docs}
# 			for d1_id, score_1 in rel_docs:
# 				# sample random doc

# 				# if sample random  
# 				rand = random.random() > 0.5
# 				# take from negative samples in ranking results file if some left and sample_random=False
# 				if len(non_rel_docs) > 0 and (rand or not self.sample_random):
# 					d2_id, d2_score = non_rel_docs.pop()
# 					content_2 = self.documents.get_tokenized_element(d2_id)
# 					result_2 = (d2_id, d2_score, content_2)

# 				else:
# 					#otherwise take random doc
# 					result_2 = self.sample_negative_document_result(rel_docs_set)
# 				# after tokenizing / preprocessing, some queries/documents have empty content.
# 				# If any of these are among 3 the selected ones, then we do not create this triplet sample
# 				# In that case, we return None, as soon as possible, so that other file reading operations can be avoided

# 				# retrieve tokenized content, given id
# 				query = self.queries.get_tokenized_element(q_id)
# 				content_1 = self.documents.get_tokenized_element(d1_id)
# 				result_1 = (d1_id, score_1, content_1)
# 				if result_1[2] is None or result_2[2] is None:
# 					yield None
# 					continue
# 				# randomly swap positions
# 				if random.random() > 0.5:
# 					temp = result_1
# 					result_1 = result_2
# 					result_2 = temp

# 				target = self.target_function(result_1, result_2)
# 				yield [query, result_1[2], result_2[2]], target

# class MSMarcoTrain(data.Dataset):
class TrainingTriplets(data.Dataset):
	def __init__(self, triplets_path, id2doc, id2query, max_query_len = 64, max_complete_length = 510, max_doc_len = None):
		# "open" triplets file
		self.triplets = FileInterface(triplets_path)
		if isinstance(id2query, FileInterface):
			self.id2query = id2query
		else:
			self.id2query = FileInterface(id2query)

		if isinstance(id2doc, FileInterface):
			self.id2doc = id2doc
		else:
			self.id2doc = FileInterface(id2doc)

		self.max_query_len = max_query_len
		if max_doc_len is not None:
			self.max_doc_len = max_doc_len
		else:
			self.max_doc_len = max_complete_length - max_query_len if max_complete_length != None else None

	def __len__(self):
		return len(self.triplets)

	def __getitem__(self, index):

		q_id, d1_id, d2_id = self.triplets.get_triplet(index)

		query = self.id2query.get_tokenized_element(q_id)
		doc1 = self.id2doc.get_tokenized_element(d1_id)
		doc2 = self.id2doc.get_tokenized_element(d2_id)
		# truncating queries and documents:
		query = query if self.max_query_len is None else query[:self.max_query_len]
		doc1 = doc1 if self.max_doc_len is None else doc1[:self.max_doc_len]
		doc2 = doc2 if self.max_doc_len is None else doc2[:self.max_doc_len]


		if random.random() > 0.5:
			return [query, doc1, doc2], 1
		else:
			return [query, doc2, doc1], -1


class RankingResultsTest:

	def __init__(self, ranking_results, id2query, id2doc, batch_size, min_len=5, indices=None, max_query_len = -1, max_complete_length = -1, max_doc_len = None, rerank_top_N = -1):
		# open file
		self.batch_size = batch_size

		self.ranking_results = open(ranking_results, 'r')
		self.min_len = min_len
		self.indices = indices
		if isinstance(id2query, FileInterface):
			self.id2query = id2query
		else:
			self.id2query = FileInterface(id2query)

		if isinstance(id2doc, FileInterface):
			self.id2doc = id2doc
		else:
			self.id2doc = FileInterface(id2doc)

		self.stop = False
		self.index = 0

		self.max_query_len = max_query_len
		if max_doc_len is not None:
			self.max_doc_len = max_doc_len
		else:
			self.max_doc_len = max_complete_length - max_query_len if max_complete_length != None else None

		self.rerank_top_N = rerank_top_N

	def reset(self):
		self.ranking_results.seek(0)
		self.line = None
		self.index = 0
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

		self.stop = False
		file_pos = self.ranking_results.tell()
		line = self.ranking_results.readline()
		if len(line) < 1:
			print('Read empty line, assuming file end.')
			return
		curr_q_id, _ = self.get_id(line)
		self.ranking_results.seek(file_pos)
		started_query = False

		# keeping count of documents that refer to the current query
		doc_count = 0

		while not self.stop:
			# read a number of lines equal to batch_size
			d_batch_ids = []
			d_batch_data = []


			while len(d_batch_ids) < self.batch_size:
				prev_q_id = curr_q_id
				file_pos = self.ranking_results.tell()
				line = self.ranking_results.readline()
				if len(line) < 1:
					break
				curr_q_id, doc_id = self.get_id(line)

				if curr_q_id != prev_q_id:
					self.index += 1

				#print('-line', line.strip())
				#print('curr_id', curr_q_id, 'index', self.index)
				#print('prev_id', prev_q_id)
				if self.indices is not None:
					if self.index not in self.indices:		
						#print('>>', line, 'index', self.index)
						continue
				if curr_q_id != prev_q_id and started_query:
					#print('break new q_id', line)
					self.stop = True
					self.ranking_results.seek(file_pos)
					break
			
				# extracting the token_ids and creating a numpy array


				# if index of queries to load is provided check if query index is indices	
				doc = self.get_tokenized(doc_id, self.id2doc)



				if doc is not None:
					# truncate document data 
					if self.max_doc_len is not None:
						doc = doc[:self.max_doc_len]



					d_batch_ids.append(doc_id)
					d_batch_data.append(torch.IntTensor(doc))
					#print('+', line)
					started_query = True
					q_id = [curr_q_id]



			if len(d_batch_data) < 1:
				print('Empty batch!')
				return



			query = self.get_tokenized(q_id[-1], self.id2query)	
			# truncate query
			if self.max_query_len is not None:
				query = query[:self.max_query_len]
			
			q_data = [torch.IntTensor(query)]
			d_batch_lengths = torch.FloatTensor([len(d) for d in d_batch_data])
			q_length = torch.FloatTensor([len(q) for q in q_data])

			if self.rerank_top_N != -1:


			#padd data along axis 1
			d_batch_data = pad_sequence(d_batch_data,1).long()
			q_data = padd_tensor(q_data, d_batch_data.shape[1]).long()
			yield q_id, q_data, q_length, d_batch_ids, d_batch_data, d_batch_lengths

	def batch_generator_bert_interaction(self):

		self.stop = False
		file_pos = self.ranking_results.tell()
		line = self.ranking_results.readline()
		if len(line) < 1:
			print('Read empty line, assuming file end.')
			return
		curr_q_id, _ = self.get_id(line)
		self.ranking_results.seek(file_pos)
		started_query = False
		while not self.stop:
			# read a number of lines equal to batch_size
			d_batch_ids = []
			d_batch_data = []
			while len(d_batch_ids) < self.batch_size:
				prev_q_id = curr_q_id
				file_pos = self.ranking_results.tell()
				line = self.ranking_results.readline()
				if len(line) < 1:
					break
				curr_q_id, doc_id = self.get_id(line)

				if curr_q_id != prev_q_id:
					self.index += 1

				if self.indices is not None:
					if self.index not in self.indices:		
						#print('>>', line, 'index', self.index)
						continue
				if curr_q_id != prev_q_id and started_query:
					#print('break new q_id', line)
					self.stop = True
					self.ranking_results.seek(file_pos)
					break
			
				# extracting the token_ids and creating a numpy array


				# if index of queries to load is provided check if query index is indices	
				# doc = self.get_tokenized(doc_id, self.id2doc)

				doc_data = self.id2doc.get_tokenized_element(doc_id)
				# truncate document data 
				if self.max_doc_len != None:
					doc_data = doc_data[:self.max_doc_len]

				if doc_data is not None:
					d_batch_ids.append(doc_id)
					d_batch_data.append(doc_data)
					#print('+', line)
					started_query = True
					q_id = [curr_q_id]
					q_data = self.id2query.get_tokenized_element(curr_q_id)
					# truncate query
					if self.max_query_len != None:
						q_data = q_data[:self.max_query_len]


			if len(d_batch_data) < 1:
				# print('Empty batch!')
				return


			# at this point we have :
			# q_id : [ query_id ]
			# q_data: numpy array of the query token ids.

			# lists of document ids and document content that do not exceed the batch size
			# d_batch_ids : list of document ids
			# d_batch_data : list of numpy arrays that contain the token ids of the documents

			# need to create final form of bert input + attention masks + token type ids

			batch_input_ids = []
			batch_attention_masks = []
			batch_token_type_ids = []

			# creating batch:
			for doc_data in d_batch_data:
				input_ids, attention_masks, token_type_ids = create_bert_inretaction_input(q_data, doc_data)

				batch_input_ids.append(input_ids)
				batch_attention_masks.append(attention_masks)
				batch_token_type_ids.append(token_type_ids)


			# the output should be in the form : [q_id], [doc_ids], input_ids, attention_masks, token_type_ids

			batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
			batch_attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=False)
			batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True, padding_value=0)

			yield q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids


class WeakSupervision(IterableDataset):
	def __init__(self, weak_results_fi, documents_fi, queries_fi, top_k_per_query=-1, sampler = 'uniform', target='binary',
			samples_per_query = -1, single_sample = False, shuffle = True, min_results = 2, strong_negatives = True, indices_to_use = None,
			sample_j = False, sample_random = False, max_length = -1):

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
		# top-N sampling methods, refer to uniform probability to the top N candidates and very small probability to the rest of the canidates
		elif "top-" in sampler:
			N = int(sampler.split('-')[1])
			self.sample_weights = np.asarray([1 for i in range(N)] + [0.0001 for i in range(self.max_candidates - N)])
			self.sampler_function = self.sample_top_n_probabilistically
		else:
			raise ValueError("Param 'sampler' of WeakSupervision, was not among {'top_n', 'uniform', 'zipf', 'linear', 'top-\{INTEGER}'}, but :" + str( sampler))
		# having a calculated list of indices, that will be used while sampling
		self.candidate_indices = list(range(self.max_candidates))

		# this will be used later in case suffle is True
		if indices_to_use is None:
			self.query_indices = list(range(len(self.weak_results_file)))
		else:
			self.query_indices = indices_to_use

		# whether to also generate triplets using a relevand and a random strong negative document from corpus
		self.sample_random = sample_random

		self.max_length = max_length

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

		if self.max_length != -1:
			query = query[:self.max_length]
			doc1 = doc1[:self.max_length]
			doc2 = doc2[:self.max_length]

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

				if self.sample_random:

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

							if self.sample_random:
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

	# top-N sampling methods, refer to uniform probability to the top N candidates and very small probability to the rest of the canidates
	def sample_top_n_probabilistically(self, scores_list, n, return_indices = False):
		length = len(scores_list)
		indices = self.candidate_indices[:length]
		# normalize sampling probabilities depending on the number of candidates
		p = self.sample_weights[:length] / sum(self.sample_weights[:length])
		sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
		if return_indices:
			return sampled_indices
		return [scores_list[i] for i in sampled_indices]




class MSMarcoLM(data.Dataset):

	def __init__(self, data_path, documents_path, queries_path, max_length=512):


		self.data = open(data_path, 'r').readlines()
		# subtract 3 for the special tokens that are added
		self.max_length = max_length - 3
		# "open" documents file
		self.documents = FileInterface(documents_path)
		# "open" queries file
		self.queries = FileInterface(queries_path)

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	def __len__(self):
		return len(self.data)

	def get_content(self, line):
		delim_pos = line.find('\t')
		return line[delim_pos+1:]

	def __getitem__(self, index):
		q_id, _, d1_id, _  = self.data[index].split(' ')
		query = self.get_content(self.queries.get(q_id))
		doc = self.get_content(self.get_content(self.documents.get(d1_id)))
		query = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(query))
		doc = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(doc))
		len_doc = self.max_length - len(query)
		cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
		sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
		inp = [cls_id] + query + [sep_id] + doc[:len_doc] + [sep_id]
		inp = torch.LongTensor(inp)
		return inp




def get_data_loaders_msmarco(cfg):

	if cfg.debug:
		triples = cfg.msmarco_triplets_train_debug
	else:
		triples = cfg.msmarco_triplets_train

	dataloaders = {}
	dataset = TrainingTriplets(triples, cfg.msmarco_docs_train, cfg.msmarco_query_train, \
		max_query_len = cfg.msmarco.max_query_len, max_complete_length = cfg.msmarco.max_complete_length)

	train_dataset, validation_dataset = split_dataset(train_val_ratio=0.9, dataset=dataset)


	if cfg.model_type == "bert-interaction" or cfg.model_type == "bert-interaction_pair_wise":
		collate_fn = collate_fn_bert_interaction
	else:
		collate_fn = collate_fn_padd_triples


	dataloaders['train'] = DataLoader(train_dataset,
	                                  batch_size=cfg.batch_size_train, collate_fn=collate_fn, shuffle=True, num_workers = cfg.num_workers)
	dataloaders['val'] = DataLoader(validation_dataset,
	                                  batch_size=cfg.batch_size_train, collate_fn=collate_fn, shuffle=False, num_workers = cfg.num_workers)


	sequential_num_workers = 1 if cfg.num_workers > 0 else 0


	dataloaders['test'] = RankingResultsTest(cfg.msmarco_ranking_results_test, cfg.msmarco_query_test, cfg.msmarco_docs_test, \
				cfg.batch_size_test, max_query_len = cfg.msmarco.max_query_len, max_complete_length = cfg.msmarco.max_complete_length)

	return dataloaders


def get_data_loaders_robust(cfg):
	if cfg.debug:
		ranking_results_train = cfg.robust_ranking_results_train_debug
	else:
		ranking_results_train = cfg.robust_ranking_results_train

	docs_fi = FileInterface(cfg.robust_docs)
	queries_fi = FileInterface(cfg.robust_query_train)
	dataloaders = {}

	test_queries_fi = FileInterface(cfg.robust_query_test)


	sequential_num_workers = 1 if cfg.num_workers > 0 else 0

	if cfg.provided_triplets:

		if cfg.weak_overfitting_test:
			queries_fi = test_queries_fi

		triplets_train_path = cfg.robust_triplets_path + "_train"
		triplets_val_path = cfg.robust_triplets_path + "_val"

		train_dataset = TrainingTriplets(triplets_train_path, docs_fi, queries_fi, max_query_len = cfg.robust04.max_length, max_doc_len = cfg.robust04.max_length)
		validation_dataset = TrainingTriplets(triplets_val_path, docs_fi, queries_fi, max_query_len = cfg.robust04.max_length, max_doc_len = cfg.robust04.max_length)

		dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, shuffle=True, num_workers = sequential_num_workers)
		dataloaders['val'] = DataLoader(validation_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, shuffle=True,  num_workers = sequential_num_workers)



	else:


		weak_results_fi = FileInterface(ranking_results_train)

		if cfg.weak_overfitting_test:
			# using all given queries both for training and validating
			indices_train = None
			indices_val = None
		else:
			# calculate train and validation size according to train_val_ratio
			dataset_len = offset_dict_len(ranking_results_train)
			indices_train, indices_val = split_by_len(dataset_len, ratio = 0.9)

		train_dataset = WeakSupervision(weak_results_fi, docs_fi, queries_fi, sampler = cfg.sampler, target=cfg.target, single_sample=cfg.single_sample,
						 shuffle=True, indices_to_use = indices_train, samples_per_query = cfg.samples_per_query, sample_j = cfg.sample_j, min_results=cfg.weak_min_results,
						 sample_random = cfg.sample_random, top_k_per_query = cfg.top_k_per_query, max_length = cfg.robust04.max_length)

		# if requested to validate on the weak results of the test set, then we change the FileInterface parameter values, and to be used all weak ranking results (indices)
		if cfg.validate_on_weak_test_results:
			queries_fi = test_queries_fi
			weak_results_fi = FileInterface(cfg.robust_ranking_results_test)
			indices_val = None

		validation_dataset = WeakSupervision(weak_results_fi, docs_fi, queries_fi, sampler = 'uniform', target=cfg.target, single_sample = True,
						shuffle=False, indices_to_use = indices_val, samples_per_query = cfg.samples_per_query, min_results=cfg.weak_min_results,
						sample_random = cfg.sample_random, top_k_per_query = cfg.top_k_per_query, max_length = cfg.robust04.max_length)


		dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, num_workers = sequential_num_workers)
		dataloaders['val'] = DataLoader(validation_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples,  num_workers = sequential_num_workers)


	dataloaders['test'] = RankingResultsTest(cfg.robust_ranking_results_test, test_queries_fi, docs_fi, cfg.batch_size_test, max_query_len = cfg.robust04.max_length, max_doc_len = cfg.robust04.max_length)

	return dataloaders

def get_data_loaders_robust_strong(cfg, indices_test, docs_fi, query_fi, ranking_results, max_q_len, max_d_len):


	dataloaders = {}

	#indices_test = indices_train
	sequential_num_workers = 1 if cfg.num_workers > 0 else 0
	#dataloaders['train'] = DataLoader(StrongData(ranking_results_fi, docs_fi, query_fi, indices=indices_train, target=cfg.target, sample_random=sample_random), batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, num_workers = sequential_num_workers)

	train_dataset = TrainingTriplets(ranking_results, docs_fi, query_fi, max_query_len = max_q_len, max_doc_len = max_d_len)

	
	dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size_train, collate_fn=collate_fn_padd_triples, num_workers = sequential_num_workers, shuffle=True)
	dataloaders['test'] = RankingResultsTest(cfg.robust_ranking_results_test, query_fi,  docs_fi, cfg.batch_size_test, indices=indices_test, max_query_len = max_q_len, max_doc_len = max_d_len)
	return dataloaders
