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

from transformers import BertTokenizer
from nltk import word_tokenize
import pickle


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
	def __init__(self, fname, batch_size, word2index_path, embedding, is_query, max_len=150):

		# open file
		self.batch_size = batch_size
		self.fname = fname
		self.is_query = is_query
		self.tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
		self.word2idx = pickle.load(open(word2index_path, 'rb'))
		self.max_len = max_len
		self.embedding = embedding
		self.file = None

	def reset(self):
		self.file_ = open(self.fname, 'r')

	def tokenize(self, text):
		if self.embedding == 'bert':
			tokenized_ids = self.tokenizer_bert.encode(text, max_length = self.max_len)
		elif self.embedding == 'glove':
			tokenized_ids = list()
			for word in word_tokenize(line)[:self.max_len]:
				try:
					tokenized_ids.append(self.word2idx[word.lower()])
				except:
					pass
		else:
			raise ValueError(f" Embedding {selfembedding} not valid!")
		if len(tokenized_ids) < 5:
			tokenized_ids += [0]*(5-len(tokenized_ids))


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

		while line:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while (line and ( len(batch_ids) < self.batch_size) ):
				
				id_ = self.get_id(line, self.is_query)
				curr_q_id = self.get_id(line, is_query=True)
				if curr_q_id != prev_q_id:
					prev_q_id = curr_q_id
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



def get_data_loaders(triplets_train_file, docs_file_train, query_file_train, query_file_val,
 	docs_file_val, batch_size, debug=False):

	dataloaders = {}
	dataloaders['train'] = DataLoader(MSMarco('train', triplets_train_file, docs_file_train, query_file_train, debug=debug),
	batch_size=batch_size, collate_fn=collate_fn_padd, shuffle=True)

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
