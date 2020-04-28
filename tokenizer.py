import argparse
import os
from nltk import word_tokenize
import pickle
from transformers import BertTokenizer, BertTokenizerFast






class Tokenizer():

	def __init__(self, tokenizer="bert", max_len=-1, stopwords="none", remove_unk = False, word2index_path = "data/embeddings/glove.6B.300d_word2idx_dict.p",
					lower_case=True, unk_words_filename = None):
		"""
		Stopwords:
			"none": Not removing any stopwords
			"lucene": Remove the default Lucene stopwords
			"some/path/file": each stopword is in one line, in lower case in that txt file
		"""


		if tokenizer != "bert" and tokenizer != "glove":
			raise ValueError("'tokenizer' param not among {bert/glove} !")


		self.lower = lower_case
		self.tokenizer = tokenizer

		if self.tokenizer == "bert":
			self.bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

		if self.tokenizer == "glove":
			self.glove_word2idx = pickle.load(open(word2index_path, 'rb'))

		self.max_len = max_len

		self.remove_unk = remove_unk
		self.unk_words_filename = unk_words_filename

		# delete older version of the file if there exists one
		if self.unk_words_filename is not None and os.path.isfile(self.unk_words_filename):
			print("Older version of unk words file was found. It will be deleted and updated.")
			os.remove(self.unk_words_filename) 


		self.set_stopword_ids_list(stopwords = stopwords)

		self.set_unk_word(remove_unk = remove_unk)



	def set_unk_word(self, remove_unk):

		if self.tokenizer == "bert":
			self.unk_word = "[UNK]"

		if self.tokenizer == "glove":
			self.unk_word = "unk"

		self.unk_word_id = self.get_word_id(self.unk_word)

		if remove_unk:
			self.stopword_ids_list.append(self.unk_word_id)


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
			if word in self.glove_word2idx:
				return self.glove_word2idx[word]
			else:
				# if selected, the unknown words that are found, are being written to the specified file, line by line
				if self.unk_words_filename is not None:
					with open(self.unk_words_filename, "a") as myfile:
						myfile.write(word + "\n")

				return self.glove_word2idx["unk"]


	def encode(self, text):
		""" Remove stopwords, tokenize and translate to word ids for a given text
		"""
		if self.lower:
			text = text.lower()
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















def tokenize(args):

	in_fname = args.input_file

	print(in_fname)
	# add = 'glove' if args.whitespace else 'bert'
	out_fname = f'{in_fname}_{args.tokenizer}_stop_{args.stopwords}{"_remove_unk" if args.remove_unk else ""}{"_max_len_" + str(args.max_len) if args.max_len != -1 else "" }.tsv'
	print(out_fname)

	if args.log_unk:
		unk_words_filename = out_fname + "_unk_words"
	else:
		unk_words_filename = None

	tokenizer = Tokenizer(tokenizer = args.tokenizer, max_len = args.max_len, stopwords=args.stopwords, remove_unk = args.remove_unk,
							word2index_path = args.word2index_path, unk_words_filename = unk_words_filename)

	empty_ids_filename = out_fname + "_empty_ids"

	word2idx = pickle.load(open(args.word2index_path, 'rb'))
	with open(out_fname, 'w') as out_f:
		with open(in_fname, 'r') as in_f:
			with open(empty_ids_filename, 'w') as empty_ids_f:

				for count, line in enumerate(in_f):

					if count % 100000 == 0 and count != 0:
						print(f'lines read: {count}')

					spl = line.strip().split(args.delimiter, 1)
					if len(spl) < 2:
						id_ = spl[0]
						# writing ids of text that is empty before tokenization
						empty_ids_f.write(id_ + "\n")
						continue
					
					id_, text = line.strip().split(args.delimiter, 1)

					tokenized_ids = tokenizer.encode(text)

					if len(tokenized_ids) == 0:
						# writing ids of text that is empty after tokenization
						empty_ids_f.write(id_ + "\n")
						continue

					out_f.write(id_ + ' ' + ' '.join(str(t) for t in tokenized_ids) + '\n')





if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--delimiter', type=str, default='\t')
	parser.add_argument('--input_file', type=str)
	parser.add_argument('--max_len', default=-1, type=int)
	parser.add_argument('--word2index_path', type=str, default='data/embeddings/glove.6B.300d_word2idx_dict.p')
	parser.add_argument('--tokenizer', type=str, help = "{'bert','glove'}")
	parser.add_argument('--stopwords', type=str, default="none", help = "{'none','lucene', 'some/path/file'}")
	parser.add_argument('--remove_unk', action='store_true')
	parser.add_argument('--log_unk', action='store_true')
	args = parser.parse_args()


	tokenize(args)
