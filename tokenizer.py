from torchtext.data.utils import get_tokenizer
from transformers import BertTokenizer
import pickle
import numpy as np

tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer_spacy = get_tokenizer("spacy")


folder = 'data/msmarco'
file_names = ['collection.small.tsv']
delimiter = '\t'



for name in file_names:
    with open(folder + '/' + name, 'r') as f:
        data = {}
        for line in f.readlines():
            id, text = line.split(delimiter, 1)
            tokenized = [tokenizer_bert.convert_tokens_to_ids(word) for word in tokenizer_spacy(text.lower().strip())]
            data[id] = tokenized

        pickle.dump( data, open( folder + '/' + name + '.p', "wb" ) )
