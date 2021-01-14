import random
import torch
from enlp.tokenizer import Tokenizer
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
from enlp.file_interface import FileInterface, File
from enlp.utils import split_dataset

from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

class MSMarcoLM(torch.utils.data.Dataset):

	def __init__(self, data_path, documents_path, queries_path, max_len=512, tokenized=True):
		self.data = open(data_path, 'r').readlines()
		# subtract 3 for the special tokens that are added
		self.max_len = max_len - 3
		# "open" documents file
		self.documents = File(documents_path, tokenized)
		# "open" queries file
		self.queries = File(queries_path, tokenized)

		self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

	def __len__(self):
		return len(self.data)

	def get_content(self, line):
		delim_pos = line.find('\t')
		return line[delim_pos + 1:]

	def __getitem__(self, index):
		q_id, _, d1_id, _ = self.data[index].split()
		query = self.queries[q_id]
		doc = self.documents[d1_id]
		query = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(query))
		doc = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(doc))
		len_doc = self.max_len - len(query)
		cls_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
		sep_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
		inp = [cls_id] + query + [sep_id] + doc[:len_doc] + [sep_id]
		inp = torch.LongTensor(inp)
		return inp


def create_bert_inretaction_input(q, doc):
	# inputs: query and document token ids in the form of numpy array
	# creates one input in the form : ["CLS"_id, q_t1_id, ..., q_tN_id, "SEP"_id, d_t1_id, ..., d_tM_id]
	# and the return the above input_ids, along with the corresponding attention mask and token type ids

	# Hard-coding token IDS of CLS, SEP and PAD
	cls_token_id = [101]
	sep_token_id = [102]
	pad_token_id = 0

	zero = [0]
	q_token_type_ids = np.zeros(q.shape[0])
	d_token_type_ids = np.ones(doc.shape[0])

	q_d = np.concatenate([cls_token_id, q, sep_token_id, doc])
	q_d = torch.LongTensor(q_d)

	q_d_attention_mask = torch.ones(q_d.shape[0], dtype=torch.bool)

	q_d_token_type_ids = np.concatenate([zero, q_token_type_ids, zero, d_token_type_ids])
	q_d_token_type_ids = torch.LongTensor(q_d_token_type_ids)

	return q_d, q_d_attention_mask, q_d_token_type_ids


class Sequential(IterableDataset):
	def __init__(self, fname, tokenize=False, min_len=5, max_len=150, stopwords='lucene', tokenizer='glove',
				 remove_unk=True):

		# open file
		self.file_ = open(fname, 'r')
		self.tokenize = tokenize
		self.min_len = min_len
		self.tokenizer = Tokenizer(tokenizer=tokenizer, max_len=max_len, stopwords=stopwords, remove_unk=remove_unk,
								   unk_words_filename=None)

	def __iter__(self):
		for line in self.file_:
			# getting position of '\t' that separates the doc_id and the begining of the token ids
			delim_pos = line.find('\t')
			# extracting the id
			id_ = line[:delim_pos]
			# extracting the token_ids and creating a numpy array
			if self.tokenize:
				tokens_list = self.tokenizer.encode(line[delim_pos + 1:])
			else:
				tokens_list = np.fromstring(line[delim_pos + 1:], dtype=int, sep=' ')

			if len(tokens_list) < self.min_len:
				tokens_list = np.pad(tokens_list, (0, self.min_len - len(tokens_list)))

			yield [id_, tokens_list]

class SingleSequential(IterableDataset):
	def __init__(self, path, id2query, id2doc, max_query_len=64, max_complete_len=510, max_doc_len=None):
		# "open" triplets file
		self.id2query = id2query
		self.id2doc = id2doc
		self.path = path
		self.max_query_len = max_query_len
		if max_doc_len is not None:
			self.max_doc_len = max_doc_len
		else:
			self.max_doc_len = max_complete_len - max_query_len if max_complete_len != None else None

	def __iter__(self):
		with open(self.path, 'r') as file:
			for line in file:
				# getting position of '\t' that separates the doc_id and the begining of the token ids
				q_id, _, d1_id = line.strip().split()
				query = self.id2query[q_id]
				doc1 = self.id2doc[d1_id]
				# truncating queries and documents:
				query = query if self.max_query_len is None else query[:self.max_query_len]
				doc1 = doc1 if self.max_doc_len is None or doc1 is None else doc1[:self.max_doc_len]

				yield query, doc1


class TriplesSequential(IterableDataset):
	def __init__(self, triples_path, id2query, id2doc, max_query_len=64, max_complete_len=510, max_doc_len=None, rand_p=0, sample_random_docs=False):
		# "open" triplets file
		self.id2query = id2query
		self.id2doc = id2doc
		self.triples_path = triples_path
		self.max_query_len = max_query_len
		self.sample_random_docs = sample_random_docs
		self.doc_list = list(self.id2doc.file.keys())
		if max_doc_len is not None:
			self.max_doc_len = max_doc_len
		else:
			self.max_doc_len = max_complete_len - max_query_len if max_complete_len != None else None
		self.rand_p = rand_p

	def random_shuffle(self, x, p=0.5):
		b = np.random.random(len(x)) > 0.5
		c = x[b]
		np.random.shuffle(c)
		x[b] = c
		return x


	def __iter__(self):
		with open(self.triples_path, 'r') as triples:
			for line in triples:
				# getting position of '\t' that separates the doc_id and the begining of the token ids
				q_id, d1_id, d2_id = line.strip().split('\t')
				query = self.id2query[q_id]
				doc1 = self.id2doc[d1_id]
				if not self.sample_random_docs:
					doc2 = self.id2doc[d2_id]
				else:
					doc2 = self.id2doc[random.choice(self.doc_list)]
						# truncating queries and documents:
				query = query if self.max_query_len is None else query[:self.max_query_len]

				doc1 = doc1 if self.max_doc_len is None or doc1 is None else doc1[:self.max_doc_len]
				doc2 = doc2 if self.max_doc_len is None or doc2 is None else doc2[:self.max_doc_len]
				if self.rand_p > 0:
					raise NotImplementedError()
					doc1 = self.random_shuffle(doc1, self.rand_p)
					doc2 = self.random_shuffle(doc2, self.rand_p)
					query = self.random_shuffle(query, self.rand_p)

				yield query, doc1, doc2, 1
				#if random.random() > 0.5:
					#yield query, doc1, doc2, 1
				#else:
				#	yield query, doc2, doc1, -1


class RankingResultsTest:

	def __init__(self, ranking_results, id2query, id2doc, batch_size, min_len=5, indices=None, max_query_len=None,
				 max_complete_len=512, max_doc_len=None, rerank_top_N=-1, device=None):
		# open file
		self.batch_size = batch_size

		self.ranking_results = open(ranking_results, 'r')
		self.min_len = min_len
		self.indices = indices
		self.id2query = id2query
		self.id2doc = id2doc
		self.device = device
		self.stop = False

		self.max_query_len = max_query_len
		if max_doc_len is not None or max_query_len is None:
			self.max_doc_len = max_doc_len
		else:
			self.max_doc_len = max_complete_len - max_query_len

		self.rerank_top_N = rerank_top_N

	def reset(self):
		self.ranking_results.seek(0)
		self.line = None
		return self

	def padd_tensor(self, sequences, max_len):
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

	def get_id(self, line):
		spl = line.split(' ')
		q_id = str(spl[0].strip())
		d_id = str(spl[2].strip())
		return q_id, d_id

	def get_tokenized(self, id_, id2tokens):
		tokenized_ids = id2tokens[id_]
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
					print("Empty Line!!!")
					break
				curr_q_id, doc_id = self.get_id(line)

				# print('-line', line.strip())
				# print('prev_id', prev_q_id)
				if self.indices is not None:
					if curr_q_id not in self.indices:
					  #  print('>>', line)
						continue
				if curr_q_id != prev_q_id and started_query:
					# print('break new q_id', line)
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


					# ignoring everything lower than the  top N documents, regarding the reranking
					if doc_count == self.rerank_top_N:
						continue

					doc_count += 1

					d_batch_ids.append(doc_id)

					d_batch_data.append(torch.LongTensor(doc).to(self.device))
					# print('+', line)
					started_query = True
					q_id = [curr_q_id]

			if len(d_batch_data) < 1:
				print('Empty batch!')
				return

			query = self.get_tokenized(q_id[-1], self.id2query)
			# truncate query
			if self.max_query_len is not None:
				query = query[:self.max_query_len]
			q_data = [torch.LongTensor(query).to(self.device)]
			d_batch_lengths = torch.FloatTensor([len(d) for d in d_batch_data]).to(self.device)
			q_length = torch.FloatTensor([len(q) for q in q_data]).to(self.device)

			# padd data along axis 1
			d_batch_data = pad_sequence(d_batch_data, 1)
			q_data = self.padd_tensor(q_data, d_batch_data.shape[1])
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

				if self.indices is not None:
					if curr_q_id not in self.indices:
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

				doc_data = self.id2doc[doc_id]
				# truncate document data


				if doc_data is not None:

					if self.max_doc_len != None:
						doc_data = doc_data[:self.max_doc_len]
					d_batch_ids.append(doc_id)
					d_batch_data.append(doc_data)
					# print('+', line)
					started_query = True
					q_id = [curr_q_id]
					q_data = self.id2query[curr_q_id]
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

	batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_targets = list(), list(), list(), list()

	for item in batch:
		# for weak supervision datasets, some queries/documents have empty text.
		# In that case the sample is None, and we skip these samples
		if item is None:
			continue

		q, doc1, doc2, target = item

		if target == 1:
			q_d1_target = 1
			q_d2_target = 0
		else:
			q_d1_target = 0
			q_d2_target = 1

		if q is not None and doc1 is not None and doc2 is not None:
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

def collate_fn_bert_interaction_single(batch):
	batch_input_ids, batch_attention_masks, batch_token_type_ids = list(), list(), list()

	for item in batch:
		# for weak supervision datasets, some queries/documents have empty text.
		# In that case the sample is None, and we skip these samples
		if item is None:
			continue

		q, doc1 = item

		if q is not None and doc1 is not None:
			q_d1, q_d1_attention_mask, q_d1_token_type_ids = create_bert_inretaction_input(q, doc1)

			batch_input_ids.append(q_d1)

			batch_attention_masks.append(q_d1_attention_mask)

			batch_token_type_ids.append(q_d1_token_type_ids)


	# in case this batch does not contain any samples, then we return None
	if len(batch_input_ids) == 0:
		return None

	batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=0)
	batch_attention_masks = pad_sequence(batch_attention_masks, batch_first=True, padding_value=False)
	batch_token_type_ids = pad_sequence(batch_token_type_ids, batch_first=True, padding_value=0)
	return batch_input_ids, batch_attention_masks, batch_token_type_ids


def collate_fn_padd_single(batch):
	""" Collate function for aggregating samples into batch size.
		returns:
		batch_data = Torch([ id_1, tokens_2, ..., id_2, tokens_2, ... ])
		batch_lengths = length of each query/document,
			that is used for proper averaging ignoring 0 padded inputs
	"""
	# batch * [id, tokens]

	batch_lengths = list()
	batch_ids, batch_data = list(), list()

	for item in batch:
		# for weak supervision datasets, some queries/documents have empty text.
		# In that case the sample is None, and we skip this samples
		if item is None:
			continue

		id_, tokens = item
		batch_data.append(torch.LongTensor(tokens))
		batch_ids.append(id_)

	# in case this batch does not contain any samples, then we return None
	if len(batch_data) == 0:
		return None

	batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
	# pad data along axis 1
	batch_data = pad_sequence(batch_data, 1)
	return batch_ids, batch_data, batch_lengths


def collate_fn_padd_triples(batch):
	""" Collate function for aggregating samples into batch size.
		returns:

	"""

	batch_q, batch_doc1, batch_doc2, batch_targets = list(), list(), list(), list()

	for i, item in enumerate(batch):

		if item is None:
			continue

		q, doc1, doc2, target = item

		if doc1 is not None and doc2 is not None:
			batch_q.append(torch.LongTensor(q))
			batch_doc1.append(torch.LongTensor(doc1))
			batch_doc2.append(torch.LongTensor(doc2))
			batch_targets.append(target)

	if len(batch_targets) == 0:
		return None

	batch_targets = torch.Tensor(batch_targets)

	batch_lengths_q = torch.FloatTensor([len(q) for q in batch_q])
	batch_lengths_doc1 = torch.FloatTensor([len(d) for d in batch_doc1])
	batch_lengths_doc2 = torch.FloatTensor([len(d) for d in batch_doc2])
	# pad data along axis 1


	batch_q = pad_sequence(batch_q, 1)
	batch_doc1 = pad_sequence(batch_doc1, 1)
	batch_doc2 = pad_sequence(batch_doc2, 1)

	return batch_q, batch_doc1, batch_doc2, batch_lengths_q, batch_lengths_doc1, batch_lengths_doc2, batch_targets


def get_data_loaders_msmarco(cfg, device=None):

	query_fi_train = File(cfg.msmarco_query_train)
	docs_fi_train = File(cfg.msmarco_docs_train)
	dataloaders = {}


	triples_train_path = cfg.msmarco_triplets + "_train"
	triples_val_path = cfg.msmarco_triplets + "_val"

	train_dataset = TriplesSequential(triples_train_path, query_fi_train, docs_fi_train,
								max_query_len=cfg.msmarco.max_query_len,
								max_complete_len=cfg.msmarco.max_complete_len, rand_p=cfg.rand_p)

	validation_dataset = TriplesSequential(triples_val_path, query_fi_train, docs_fi_train,
								max_query_len=cfg.msmarco.max_query_len,
								max_complete_len=cfg.msmarco.max_complete_len)

	if cfg.model_type == "bert-interaction":
		collate_fn = collate_fn_bert_interaction
	else:
		collate_fn = collate_fn_padd_triples

	if cfg.sub_batch_size != None:
		cfg.batch_size_train = cfg.sub_batch_size

	dataloaders['train'] = DataLoader(train_dataset,
									  batch_size=cfg.batch_size_train, collate_fn=collate_fn,
									  num_workers=cfg.num_workers)
	dataloaders['val'] = DataLoader(validation_dataset,
									batch_size=cfg.batch_size_train ,collate_fn=collate_fn,
									num_workers=cfg.num_workers)

	queries_fi_test = File(cfg.msmarco_query_test)
	docs_fi_test = File(cfg.msmarco_docs_test)

	dataloaders['test'] = RankingResultsTest(cfg.msmarco_ranking_results_test, queries_fi_test, docs_fi_test,
											 cfg.batch_size_test, rerank_top_N=cfg.rerank_top_N,
											 max_query_len=cfg.msmarco.max_query_len,
											 max_complete_len=cfg.msmarco.max_complete_len, device=device)

	return dataloaders


def get_data_loaders_robust(cfg, device=None):

	docs_fi = File(cfg.robust_docs)
	query_fi_train= File(cfg.robust_query_train)
	query_fi_test = File(cfg.robust_query_test)
	dataloaders = {}


	sequential_num_workers = 1 if cfg.num_workers > 0 else 0


	if cfg.model_type == "bert-interaction":
		collate_fn = collate_fn_bert_interaction
		max_query_len = cfg.robust04.max_query_len
		max_complete_len = cfg.robust04.max_complete_len
		max_doc_len = None

	else:
		collate_fn = collate_fn_padd_triples
		max_query_len = cfg.robust04.max_len
		max_doc_len = cfg.robust04.max_len
		max_complete_len = None

	triples_train_path = cfg.robust_triples + "_train"
	triples_val_path = cfg.robust_triples + "_val"

	train_dataset = TriplesSequential(triples_train_path, query_fi_train, docs_fi, max_query_len=max_query_len, max_complete_len=max_complete_len, max_doc_len=max_doc_len)
	validation_dataset = TriplesSequential(triples_val_path, query_fi_train, docs_fi, max_query_len=max_query_len, max_doc_len=max_doc_len, max_complete_len=max_complete_len)



	if cfg.sub_batch_size != None:
		cfg.batch_size_train = cfg.sub_batch_size

	dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size_train,
									  collate_fn=collate_fn,num_workers=sequential_num_workers)
	dataloaders['val'] = DataLoader(validation_dataset, batch_size=cfg.batch_size_test,
									collate_fn=collate_fn, num_workers=sequential_num_workers)

	dataloaders['test'] = RankingResultsTest(cfg.robust_ranking_results_test, query_fi_test, docs_fi,
											 cfg.batch_size_test, max_query_len=max_query_len,
											 max_doc_len=max_doc_len, rerank_top_N=cfg.rerank_top_N, device=device)

	return dataloaders


def get_data_loaders_robust_strong(cfg, indices_test, query_fi, docs_fi, ranking_results, device=None):
	dataloaders = {}
	sequential_num_workers = 1 if cfg.num_workers > 0 else 0

	if cfg.sub_batch_size != None:
		cfg.batch_size_train = cfg.sub_batch_size
	print(cfg.batch_size_train)
	if cfg.model_type == "bert-interaction":
		collate_fn = collate_fn_bert_interaction
		max_query_len = cfg.robust04.max_query_len
		max_complete_len = cfg.robust04.max_complete_len
		max_doc_len = None

	else:
		collate_fn = collate_fn_padd_triples
		max_query_len = cfg.robust04.max_len
		max_doc_len = cfg.robust04.max_len
		max_complete_len = None

	train_dataset = TriplesSequential(ranking_results, query_fi, docs_fi, max_query_len=max_query_len, max_complete_len=max_complete_len, max_doc_len=max_doc_len, sample_random_docs=cfg.sample_random_docs)


	dataloaders['train'] = DataLoader(train_dataset, batch_size=cfg.batch_size_train,
									  collate_fn=collate_fn, num_workers=sequential_num_workers)
	dataloaders['test'] = RankingResultsTest(cfg.robust_ranking_results_test, query_fi, docs_fi, cfg.batch_size_test,
											 indices=indices_test, max_query_len=max_query_len, max_doc_len=max_doc_len, max_complete_len=max_complete_len,
											 rerank_top_N=cfg.rerank_top_N, device=device)
	return dataloaders
