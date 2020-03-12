from transformers import BertTokenizer
import argparse
import os
from nltk import word_tokenize
import pickle
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')





parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default='\t')
parser.add_argument('--folder', type=str)
parser.add_argument('--fname', type=str)
parser.add_argument('--max_len', type=int)
parser.add_argument('--word2index_path', type=str, default='../data/embeddings/glove.6B.300d_word2idx_dict.p')
parser.add_argument('--whitespace', action='store_true')
args = parser.parse_args()

in_fname = args.folder + args.fname
base = os.path.splitext(in_fname)[0]
add = '.white' if args.whitespace else ''
out_fname = f'{base}.tokenized{add}.tsv'

word2idx = pickle.load(open(args.word2index_path, 'rb')) 
with open(out_fname, 'w') as out_f:
	with open(in_fname, 'r') as in_f:
		#ids = list()
		count = 0
		data = {}

		
		line = in_f.readline()
		while line:

			if count % 10000 == 0:
				print(f'lines read: {count}')
			id_, text = line.strip().split(args.delimiter, 1)
			if not args.whitespace:
				tokenized_ids = tokenizer_bert.encode(text, max_length = args.max_len)
			else:
				tokenized_ids = list()
				for word in word_tokenize(line)[:args.max_len]:
					try:
						tokenized_ids.append(word2idx[word.lower()])
					except:
						pass
			out_f.write(id_ + ' ' + ' '.join(str(t) for t in tokenized_ids) + '\n')
			count += 1
			line = in_f.readline()
