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
		self.file = open(fname, 'r')
		self.batch_size = batch_size

		self.previous_line = self.file.readline()

	def batch_generator(self):

		# line = self.file.readline()
		while self.previous_line:
			# read a number of lines equal to batch_size
			batch_ids = []
			batch_data = []
			while(self.previous_line and ( len(batch_ids) < self.batch_size) ):

				# getting position of first ' ' that separates the doc_id and the begining of the token ids
				delim_pos = self.previous_line.find(' ')
				# extracting the id
				id = self.previous_line[:delim_pos]
				# extracting the token_ids and creating a numpy array
				tokens_list = np.fromstring(self.previous_line[delim_pos+1:], dtype=int, sep=' ')
				batch_ids.append(id)
				batch_data.append(torch.IntTensor(tokens_list))

				self.previous_line = self.file.readline()

			batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
			#padd data along axis 1
			batch_data = pad_sequence(batch_data,1).long()

			yield batch_ids, batch_data, batch_lengths


def get_data_loaders(triplets_train_file, docs_file_train, query_file_train, query_file_val,
 	docs_file_val, batch_size, debug=False):

	dataloaders = {}
	dataloaders['train'] = DataLoader(MSMarco('train', triplets_train_file, docs_file_train, query_file_train, debug=debug),
	batch_size=batch_size, collate_fn=collate_fn_padd, shuffle=True)

	query_batch_generator = MSMarcoSequential(query_file_val, batch_size).batch_generator()
	docs_batch_generator = MSMarcoSequential(docs_file_val, batch_size).batch_generator()

	dataloaders['val'] = [query_batch_generator, docs_batch_generator]

	#dataloaders['test'] =  DataLoader(MSMarco('eval'), batch_size=batch_size, collate_fn=collate_fn_padd)

	return dataloaders
