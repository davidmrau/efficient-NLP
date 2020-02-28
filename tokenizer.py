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

out_folder = args.folder + args.fname.replace('.tsv', '') + '/'
os.makedirs(out_folder, exist_ok=True)



with open(args.folder + args.fname, 'r') as f:
    ids = list()
    count = 0
    data = {}

    line = f.readline()
    while line:
        line = f.readline()

        if count % 10000 == 0:
            print(f'lines read: {count}')
        id, text = line.split(args.delimiter, 1)
        tokenized_ids = tokenizer_bert.encode(text)
        data[id] = tokenized_ids[:args.max_len]
        ids.append(id)

        count += 1
    out_name =  'bert.p'
    pickle.dump(data, open(out_folder + args.fname + out_name, 'wb'))
    pickle.dump(ids, open(out_folder + out_name + '.ids', 'wb' ))
