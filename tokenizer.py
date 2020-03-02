from transformers import BertTokenizer
import pickle
import numpy as np
import argparse
import os


tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')





parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default='\t')
parser.add_argument('--folder', type=str)
parser.add_argument('--fname', type=str)
parser.add_argument('--max_len', type=int)
args = parser.parse_args()


with open(args.folder + args.fname, 'r') as f:
    #ids = list()
    count = 0
    data = {}

    line = 'f'
    while line:
        line = f.readline()

        if count % 10000 == 0:
            print(f'lines read: {count}')
        if line != '':
            id_, text = line.split(args.delimiter, 1)
            tokenized_ids = tokenizer_bert.encode(text, max_length = args.max_len)
            data[id_] = tokenized_ids
            #ids.append(id_)

        count += 1
    pickle.dump(data, open(args.folder + args.fname + '.p', 'wb'))
    #pickle.dump(ids, open(args.folder + args.fname + 'ids.p', 'wb' ))
