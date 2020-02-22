
import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path

class MSMarco(data.Dataset):

    def __init__(self, dataset_path, split):


        self.triplets = read_triplets(f'{dataset_path}/qidpidtriples.train.full.tsv')
        self.docs_path = path.join(dataset_path, 'collection.small/')
        self.queries_path = path.join(dataset_path, f'queries.{split}/')
        self.doc_ids = np.load(f'{self.docs_path}ids.npy')


    def __len__(self):
        # double size, because dataset contain only relevant examples
        return self.triplets.shape[0]*2

    def __getitem__(self, index):
        # get relevant example
        q_id, d1_id, d2_id = self.triplets[index - int(len(self)/2)]
        # either stay with the relevant doc2 or sample a random doc2
        if index <= len(self)/2:
            target = 1
        else:
            #d2_id = np.random.choice(self.doc_ids)
            target = -1


        query = np.load(f'{self.docs_path}{q_id}.npy')
        doc1 = np.load(f'{self.docs_path}{d1_id}.npy')
        doc2 = np.load(f'{self.docs_path}{d2_id}.npy')

        return [query, doc1, doc2], target
