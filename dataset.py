import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path
from fake_data import *
from random import randint
from file_interface import FileInterface



class MSMarco(data.Dataset):

	def __init__(self, dataset_path, split,  debug=False):

		self.split = split

		if split == 'train':
			debug_str = '' if not debug else '.debug'
			triplets_fname = f'{dataset_path}/qidpidtriples.{split}.full{debug_str}.tsv'
			# "open" triplets file
			self.triplets = FileInterface(triplets_fname)


		# "open" documents file
		self.documents = FileInterface(f'{dataset_path}/collection.tokenized.tsv')
		# "open" queries file
		self.queries = FileInterface(f'{dataset_path}/queries.{split}.tokenized.tsv')
		# read qrels
		self.qrels = read_qrels(f'{dataset_path}/qrels.{split}.tsv')

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

			q_id, d1_id = self.qrels[index]

			d2_id = randint(0, self.max_doc_id)
			# making sure that the sampled d1_id is diferent from relevant d2_id
			while int(d1_id) == d2_id:
				d2_id = randint(0, self.max_doc_id)
			# type cast the id into a string
			d2_id = str(d2_id)
		else:
			raise ValueError(f'Unknown split: {split}')

		query = self.queries.get_tokenized_element(q_id)
		doc1 = self.documents.get_tokenized_element(d1_id)
		doc2 = self.documents.get_tokenized_element(d2_id)

		if np.random.random() > 0.5:
			return [query, doc1, doc2], 1
		else:
			return [query, doc2, doc1], -1


class MSMarcoInference(data.Dataset):
	def __init__(self, fname):
		self.file = open(fname, 'r')
	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		id_, data = self.data.get_tokenized_element(index)
		return [data], id_
