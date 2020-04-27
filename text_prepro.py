from transformers import BertTokenizer, BertTokenizerFast
import argparse
import os
from nltk import word_tokenize
import pickle


class Tokenizer():

	def __init__(self, tokenizer="bert", max_len=-1, stopwords="none", remove_unk = False, word2index_path = "data/embeddings/glove.6B.300d_word2idx_dict.p"):
		"""
		Stopwords:
			"none": Not removing any stopwords
			"lucene": Remove the default Lucene stopwords
			"some/path/file": each stopword is in one line, in lower case in that txt file
		"""


		if tokenizer != "bert" and tokenizer != "glove":
			raise ValueError("'tokenizer' param not among {bert/glove} !")



		# self.tokenizer = tokenizer

		if self.tokenizer == "bert":
			self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

		if self.tokenizer == "glove":
			self.glove_word2idx = pickle.load(open(word2index_path, 'rb'))

		self.max_len = max_len

		self.remove_unk = remove_unk

		self.set_stopword_ids_list(stopwords = stopwords)

		if self.remove_unk:
			self.add_unknown_to_stopword_list_ids()

	def add_unknown_to_stopword_list_ids(self):

		if self.tokenizer == "bert":
			unk_word = "[UNK]"

		if self.tokenizer == "glove":
			unk_word = "unk"

		unk_word_id = self.get_word_id(unk_word)

		self.stopword_ids_list.append(unk_word_id)


	def set_stopword_ids_list(self, stopwords):
		if stopwords == "none":
			self.stopword_ids_list = []

		elif stopwords == "lucene":
			# If not specified, using standard Lucene stopwords list

			# Lucene / anserini default stopwords:
			# https://stackoverflow.com/questions/17527741/what-is-the-default-list-of-stopwords-used-in-lucenes-stopfilter
			# The default stop words set in StandardAnalyzer and EnglishAnalyzer is from StopAnalyzer.ENGLISH_STOP_WORDS_SET:
			# https://github.com/apache/lucene-solr/blob/master/lucene/analysis/common/src/java/org/apache/lucene/analysis/en/EnglishAnalyzer.java#L46

			lucene_stopwords_list = ["a", "an", "and", "are", "as", "at", "be", "but", "by",
			"for", "if", "in", "into", "is", "it",
			"no", "not", "of", "on", "or", "such",
			"that", "the", "their", "then", "there", "these",
			"they", "this", "to", "was", "will", "with"]
			stopwords_list = lucene_stopwords_list


			self.stopword_ids_list = [self.get_word_id(word.lower()) for word in stopwords_list]


		else:
			raise ValueError("Implement function to read stopwords from provided 'stopwords' argument!")


	def get_word_id(self, word):
		if self.tokenizer == "bert":
			return self.bert_tokenizer.encode(word)[1]
		if self.tokenizer == "glove":
			if word.lower() in self.glove_word2idx:
				return self.glove_word2idx[word.lower()]
			else:
				return self.glove_word2idx["unk"]


	def encode(self, text):
		""" Remove stopwords, tokenize and translate to word ids for a given text
		"""

		if self.tokenizer == "bert":
			# removing CLS and SEP token which are added to the beginning and the end of input respectively
			temp_encoded = self.bert_tokenizer.encode(text)[1:-1]
		elif self.tokenizer == "glove":
			# tokenize
			tokens = word_tokenize(text)

			# translate to word ids
			temp_encoded = [self.get_word_id(word) for word in tokens]


		# remove stopwords
		if len(self.stopword_ids_list) != 0:
			encoded = []
			for word_id in temp_encoded:
				if word_id not in self.stopword_ids_list:
					encoded.append(word_id)
		else:
			encoded = temp_encoded

		# enforce maximum length
		if self.max_len != -1:
			encoded = encoded[:self.max_len]

		return encoded


	def decode(self, word_ids):

		if self.tokenizer == "bert":
			return self.bert_tokenizer.decode(word_ids)

		if self.tokenizer == "glove":

			# make sure that we have constracted a id2word mapping
			if not hasattr(self, "glove_idx2word"):
				# If not, then we build it once 
				self.glove_idx2word = {}
				for word in self.glove_word2idx:
					word_id = self.glove_word2idx[word]
					self.glove_idx2word[ word_id ] = word

			# translate into words in a string split by ' ' and return it
			return ' '.join(self.glove_idx2word[word_id] for word_id in word_ids)



	def get_word_from_id(self,word_id):
		pass

