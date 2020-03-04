from utils import *
import numpy as np




class FileInterface:
	""" This interface class is used for reading large files, withouth loading them on memory

		constructor params :
		filename			: str, the name/path of the large file
		seek_dict_filename	: str, the name/path of the dictionary pickle file, that contains a dictionary
			the dictionary matches element ids (doc_id / query_id / line_id for triplets) to their seek value within the file
	"""

	def __init__(self, filename):
			# check if filename exists
			try:
				# we keep the file always open
				self.file = open(filename, 'r')
			except:
				raise IOError(f'File: {filename} not accessible!')

			# check if dictionary pickle exists
			try:
					# given an id of an element, this dictionary returns the seek value of the file to the corresponding line
				self.seek_dict = read_pickle(filename + '.offset_dir.p')
			except:
				raise IOError(f'File: {filename}.offset_dir.p not accessible!\nYou need to first create the dictionary with seek values for this file!!\nCheck offset_dict.py')



	def __len__(self):
		# length of items is equal to the length of the dictionary of seek values
		return len(self.seek_dict)

	def get(self, id):
			# seek the file to the line that you want to read
			self.file.seek( self.seek_dict[id] )
			# read and return the line
			return self.file.readline()

	def get_triplet(self, index):
		# returns 3 strings q_id, d1_id, d2_id
			return self.get(index).strip().split('\t')


	def get_tokenized_element(self, id):
			""" Used to read one Document OR Query from file, given the
			"""
			# return (str) element_id, and numpy array of token ids
			line = self.get(id)
			# getting position of first ' ' that separates the doc_id and the begining of the token ids
			delim_pos = line.find(' ')
			# extracting the id
			# id = line[:delim_pos]
			# extracting the token_ids and creating a numpy array
			tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
			return tokens_list
