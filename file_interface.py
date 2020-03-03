from utils import *
import numpy as np




class FileInterface:
	""" This interface class is used for reading large files, withouth loading them on memory

		constructor params :
		filename			: str, the name/path of the large file
		seek_dict_filename	: str, the name/path of the dictionary pickle file, that contains a dictionary
			the dictionary matches element ids (doc_id / query_id / line_id for triplets) to their seek value within the file
	"""

	def __init__(self, filename, map_index=True):
			# check if filename exists
			try:
					# we keep the file always open
				self.file = open(filename, 'r')
			except:
				raise IOError(f'File: {filename} not accessible!')

			# check if dictionary pickle exists
			try:
					# given an id of an element, this dictionary returns the seek value of the file to the corresponding line
				self.seek_dict = read_pickle(filename + '.offset_list.p')
			except:
				raise IOError(f'File: {filename}.offset_list.p not accessible!\nYou need to first create the dictionary with seek values for this file!!\nCheck offset_dict.py')
			# create a mapping from index to id using a list
			if map_index:
				self.index_to_id = list(self.seek_dict.keys())
			else:
				self.index_to_id = None


	def __len__(self):
		# length of items is equal to the length of the dictionary of seek values
		return len(self.seek_dict)

	def get(self, index):

			# if ids != linecounts: mapping from index is needed
			if self.index_to_id != None:
				id = self.index_to_id[index]
			else:
				id = index

			# seek the file to the line that you want to read

			self.file.seek( self.seek_dict[id] )
			# read and return the line
			return self.file.readline()

	def get_triplet(self, index):
		# returns 3 strings q_id, d1_id, d2_id
			return self.get(index).strip().split('\t')


	def get_tokenized_element(self, index):
			""" Used to read one Document OR Query from file, given the
			"""
			# return (str) element_id, and numpy array of token ids
			line = self.get(index)
			# getting position of first ' ' that separates the doc_id and the begining of the token ids
			delim_pos = line.find(' ')
			# extracting the id
			id = line[:delim_pos]
			# extracting the token_ids and creating a numpy array
			tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
			return id, tokens_list

		#
		#
		#
		# self.split = split
		# self.debug = debug
		#
		# if split == 'train':
		# 	debug_str = '' if not debug else '.debug'
		# 	triplets_fname = f'{dataset_path}/qidpidtriples.{split}.full{debug_str}.tsv'
		# 	if not debug:
		# 		self.triplets_offset_list = read_pickle(f'{triplets_fname}.offset_list.p')
		# 		self.triplets_file = open(triplets_fname, 'r')
		# 	else:
		# 		self.triplets = read_triplets(triplets_fname)
		# self.docs_offset_list = docs_offset_list
		# self.qrels = read_qrels(path.join(dataset_path, f'qrels.{split}.tsv'))
		# self.queries = read_pickle(f'{dataset_path}/queries.{split}.tsv.p')
		#
		# self.docs_file = open(f'{dataset_path}/collection.tokenized.tsv', 'r')
		#
		# self.max_doc_id = len(self.docs_offset_list) - 1
