import multiprocessing

import indexed_gzip as igzip
import numpy as np
from enlp.utils import get_offset_dict_path, read_pickle


class FileInterface:
	""" This interface class is used for reading large files, withouth loading them on memory

		constructor params :
		filename			: str, the name/path of the large file
		seek_dict_filename	: str, the name/path of the dictionary pickle file, that contains a dictionary
			the dictionary matches element ids (doc_id / query_id / line_id for triplets) to their seek value within the file
	"""

	# A class variable in order to make file reading to be multiprocess safe
	# it is a dictionary because each filename uses different lock
	locks = {}

	def __init__(self, filename):

		self.filename = filename
		self.decode = False
		offset_dict_path = get_offset_dict_path(filename)
		# check if filename exists
		try:
			# we keep the file always open
			# if gzipped open with igzip
			if filename.endswith('.gz'):
				self.file = igzip.IndexedGzipFile(filename)
				self.decode = True
			else:
				self.file = open(filename, 'r')
		except:
			raise IOError(f'File: {filename} not accessible!')
		
		# check if dictionary pickle exists
		try:
				# given an id of an element, this dictionary returns the seek value of the file to the corresponding line
			self.seek_dict = read_pickle(offset_dict_path)
		except:
			raise IOError(f'File: {offset_dict_path} not accessible!\nYou need to first create the dictionary with seek values for this file!!\nCheck offset_dict.py')

		if self.filename in FileInterface.locks:
			raise ValueError(f'Filename "{self.filename}", is alredy open from another FileInterface instance !!!\nMultiprocessing will not work!')

		# create a multiprocessing.Lock, for this file specifically
		FileInterface.locks[self.filename] = multiprocessing.Lock()


	def __len__(self):
		# length of items is equal to the length of the dictionary of seek values
		return len(self.seek_dict)

	def get(self, id_):

		with FileInterface.locks[self.filename]:
			# seek the file to the line that you want to read
			self.file.seek( self.seek_dict[id_] )
			# read and return the line
			line = self.file.readline()
			if self.decode:
				line = line.decode()
			return line

	def get_triplet(self, index):
		# returns 3 strings q_id, d1_id, d2_id
		return self.get(index).strip().split('\t')


	def get_tokenized_element(self, id_):
		""" Used to read one Document OR Query from file, given the
		"""
		# return (str) element_id, and numpy array of token ids
		line = self.get(id_)
		# getting position of '\t' that separates the doc_id and the begining of the token ids
		delim_pos = line.find('\t')
		# in case the tokenized of the line is empy, and the line only contains the id, then we return None
		# example of line with empy text: line = "1567 \n" -> len(line[delim_pos+1:]) == 1
		if len(line[delim_pos+1:]) < 2:
			return None 
		# extracting the token_ids and creating a numpy array
		tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
		return tokens_list


	def read_all_results_of_query_index(self, q_index, top_k_per_query = -1):

		results = []

		with FileInterface.locks[self.filename]:
			# set file seek to the beginning of the line of the first result of that query
			self.file.seek( self.seek_dict[q_index] )


			line = self.file.readline()
			
			if self.decode:
				line = line.decode()

			requested_q_id = line.split()[0]

			while line and len(line) != 0:


				# decompose line
				split_line = line.split()

				q_id = split_line[0]
				doc_id = split_line[2]
				# rank = split_line[3]
				score = float(split_line[4])

				# if we started reading the results of the next query
				if q_id != requested_q_id:
					break

				results.append( (doc_id, score) )

				if top_k_per_query != -1 and len(results) == top_k_per_query:
					break

				# read next line
				line = self.file.readline()
				if self.decode:
					line = line.decode()

		return requested_q_id, results
				
