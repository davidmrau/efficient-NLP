from transformers import BertTokenizer
import argparse
import os
from nltk import word_tokenize
import pickle
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')

from text_prepro import Tokenizer





parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default='\t')
parser.add_argument('--input_file', type=str)
parser.add_argument('--max_len', default=-1, type=int)
parser.add_argument('--word2index_path', type=str, default='data/embeddings/glove.6B.300d_word2idx_dict.p')
parser.add_argument('--tokenizer', type=str, help = "{'bert','glove'}")
parser.add_argument('--stopwords', type=str, default="none", help = "{'none','lucene', 'some/path/file'}")
parser.add_argument('--remove_unk', action='store_true')
args = parser.parse_args()



tokenizer = Tokenizer(tokenizer = args.tokenizer, max_len = args.max_len, stopwords=args.stopwords, remove_unk = args.remove_unk,
						word2index_path = args.word2index_path)



in_fname = args.input_file

print(in_fname)


# add = 'glove' if args.whitespace else 'bert'
out_fname = f'{in_fname}.{args.tokenizer}.stop_{args.stopwords}{".remove_unk" if args.remove_unk else ""}.len_{args.max_len}.tsv'

print(out_fname)
word2idx = pickle.load(open(args.word2index_path, 'rb')) 
with open(out_fname, 'w') as out_f:
	with open(in_fname, 'r') as in_f:
		#ids = list()
		count = 0
		data = {}

		
		line = in_f.readline()
		while line:

			if count % 100000 == 0 and count != 0:
				print(f'lines read: {count}')


			id_, text = line.strip().split(args.delimiter, 1)

			tokenized_ids = tokenizer.encode(text)
			# if not args.whitespace:
			# 	tokenized_ids = tokenizer_bert.encode(text, max_length = args.max_len)
			# else:
			# 	tokenized_ids = list()
			# 	for word in word_tokenize(line)[:args.max_len]:
			# 		try:
			# 			tokenized_ids.append(word2idx[word.lower()])
			# 		except:
			# 			pass


			out_f.write(id_ + ' ' + ' '.join(str(t) for t in tokenized_ids) + '\n')
			count += 1
			line = in_f.readline()
