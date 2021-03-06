import csv
import re
import torch
import numpy as np
class DataReader:

	def __init__(self, data_file, num_docs,  multi_pass, id2q, id2d, VOCAB_SIZE, NUM_HIDDEN_NODES, MAX_DOC_TERMS, MAX_QUERY_TERMS, DATA_FILE_VOCAB, DATA_EMBEDDINGS, MB_SIZE):

		self.num_docs = num_docs
		self.doc_col = 2 if self.num_docs <= 1 else 1
		self.multi_pass = multi_pass
		self.regex_drop_char = re.compile('[^a-z0-9\s]+')
		self.regex_multi_space = re.compile('\s+')
		self.DATA_FILE_VOCAB = DATA_FILE_VOCAB
		self.VOCAB_SIZE = VOCAB_SIZE
		self.NUM_HIDDEN_NODES = NUM_HIDDEN_NODES
		self.MAX_DOC_TERMS = MAX_DOC_TERMS
		self.MAX_QUERY_TERMS = MAX_QUERY_TERMS
		self.DATA_EMBEDDINGS = DATA_EMBEDDINGS
		self.MB_SIZE = MB_SIZE
		self.id2d = id2d
		self.id2q = id2q
		self.__load_vocab()
		self.__init_data(data_file)
		self.__allocate_minibatch()

	def __tokenize(self, s, max_terms):
		return self.regex_multi_space.sub(' ', self.regex_drop_char.sub(' ', s.lower())).strip().split()[:max_terms]

	def __load_vocab(self):
		self.vocab = {}
		with open(self.DATA_FILE_VOCAB, mode='r', encoding="utf-8") as f:
			reader = csv.reader(f, delimiter='\t')
			for row in reader:
				self.vocab[row[0]] = int(row[1])
		self.VOCAB_SIZE = len(self.vocab) + 1
		embeddings = np.zeros((self.VOCAB_SIZE, self.NUM_HIDDEN_NODES), dtype=np.float32)
		with open(self.DATA_EMBEDDINGS, mode='r', encoding="utf-8") as f:
			for line in f:
				cols = line.split()
				idx = self.vocab.get(cols[0], 0)
				if idx > 0:
					for i in range(self.NUM_HIDDEN_NODES):
						embeddings[idx, i] = float(cols[i + 1])
		self.pre_trained_embeddings = torch.tensor(embeddings)


	def pad(self, A, length):
		arr = np.zeros(length)
		arr[:len(A)] = A
		return arr

	def __init_data(self, file_name):
		self.reader = open(file_name, mode='r', encoding="utf-8")
		print('Num docs:', self.num_docs)
		self.reader.seek(0)


	def __allocate_minibatch(self):
		self.features = {}
		self.features['d'] = []
		self.features['q'] = []
		self.features['lengths_d'] = []
		self.features['lengths_q'] = []
		for i in range(self.num_docs):
			self.features['d'].append(np.zeros((self.MB_SIZE, self.MAX_DOC_TERMS), dtype=np.int64))
			self.features['q'].append(np.zeros((self.MB_SIZE, self.MAX_QUERY_TERMS), dtype=np.int64))
			self.features['lengths_d'].append(np.zeros((self.MB_SIZE, 1), dtype=np.int64))
			self.features['lengths_q'].append(np.zeros((self.MB_SIZE, 1), dtype=np.int64))
		self.features['labels'] = np.ones((self.MB_SIZE), dtype=np.int64)
		self.features['meta'] = []

	def pad(self, A, length):
		arr = np.zeros(length)
		arr[:len(A)] = A
		return arr

	def __clear_minibatch(self):
		for i in range(self.num_docs):
			self.features['d'][i].fill(np.int64(0))
			self.features['q'][i].fill(np.int64(0))
			self.features['lengths_d'][i].fill(np.int64(0))
			self.features['lengths_q'][i].fill(np.int64(0))

		self.features['meta'].clear()
	def get_minibatch(self):
		self.__clear_minibatch()
		for i in range(self.MB_SIZE):
			row = self.reader.readline()
			if row == '':
				if self.multi_pass:
					self.reader.seek(0)
					row = self.reader.readline()
				else:
					break
			cols = row.split()
			q = self.id2q[cols[0]][:self.MAX_QUERY_TERMS]
			ds = [self.id2d[cols[self.doc_col + i].strip()] for i in range(self.num_docs)]

			ds = [ e[:self.MAX_DOC_TERMS]  if e is not None else [0] for e in ds]

			#ds = [self.__tokenize(cols[self.num_meta_cols + i + 1], self.MAX_DOC_TERMS) for i in range(self.num_docs)]
			for d in range(self.num_docs):
				self.features['d'][d][i] = self.pad(ds[d], self.MAX_DOC_TERMS)
				self.features['q'][d][i] = self.pad(q, self.MAX_QUERY_TERMS)
				self.features['lengths_q'][d][i] = len(q)
				self.features['lengths_d'][d][i] = len(ds[d])

			if self.num_docs == 1:
				self.features['meta'].append([cols[0], cols[2], float(cols[4])])
		return self.features

	def reset(self):
		self.reader.seek(0)
