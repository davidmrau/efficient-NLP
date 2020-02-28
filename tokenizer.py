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
args = parser.parse_args()

out_folder = args.folder + args.fname.replace('.tsv', '') + '/'
os.makedirs(out_folder, exist_ok=True)



with open(args.folder + args.fname, 'r') as f:
    ids = list()
    lines = f.readlines()
    count = 0
    data = {}
    for line in lines:
        if count % 10000 == 0:
            print(f'{round(count/ len(lines),5)} %')
        id, text = line.split(args.delimiter, 1)
        tokenized_id = tokenizer_bert.encode(text)
        data[id] = tokenized_id
        ids.append(id)

        count += 1
    out_name =  '.p'
    pickle.dump(data, open(out_folder + args.fname + out_name, 'wb'))
    pickle.dump(ids, open(out_folder + out_name + '.ids', 'wb' ))
