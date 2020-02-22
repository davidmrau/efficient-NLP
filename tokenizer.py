from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
import pickle
import numpy as np
import argparse
import os


tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_spacy = get_tokenizer("spacy")




parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default='\t')
parser.add_argument('--folder', type=str)
parser.add_argument('--fname', type=str)
parser.add_argument('--bert', action='store_true')
args = parser.parse_args()

out_folder = args.folder + args.fname.replace('.tsv', '') + '/'
os.makedirs(out_folder, exist_ok=True)



with open(args.folder + args.fname, 'r') as f:
    ids = list()
    lines = f.readlines()
    count = 0
    for line in lines:
        if count % 10000 == 0:
            print(f'{round(count/ len(lines),5)} %')
        id, text = line.split(args.delimiter, 1)
        
        if args.bert:
            tokenized_id = tokenizer_bert.encode(text)
        else:
            tokenized = tokenizer_spacy(text.lower().strip())
            tokenized_id = tokenizer_bert.convert_tokens_to_ids(tokenized)

        np.save(out_folder + id, tokenized_id)
        ids.append(id)

        count += 1
    id_out_name =  'bert.ids' if args.bert else 'ids'
    np.save(out_folder + id_out_name, ids )
