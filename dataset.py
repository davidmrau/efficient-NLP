import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path
from fake_data import *


class MSMarco(data.Dataset):

    def __init__(self, dataset_path, split, debug=False):

        self.split = split

        if not debug:
            if split == 'train':
                self.triplets = read_triplets(f'{dataset_path}/qidpidtriples.{split}.full.tsv')
            self.qrels = read_qrels(path.join(dataset_path, f'qrels.{split}.tsv'))
            self.docs = read_pickle(f'{dataset_path}/collection.tsv.p')
            self.queries = read_pickle(f'{dataset_path}/queries.{split}.tsv.p')
            self.doc_ids = read_pickle(f'{dataset_path}/collection.tsv.ids.p')
        else:
            self.triplets = triplets_fake
            self.docs = docs_fake
            self.queries = queries_fake
            self.qrels = qrels_fake
            self.doc_ids = doc_ids_fake

    def __len__(self):
        if self.split == 'train':
            return len(self.triplets)
        elif self.split == 'dev':
            return len(self.qrels)


    def __getitem__(self, index):

        if self.split == 'train':
            q_id, d1_id, d2_id = self.triplets[index]
        elif self.split == 'dev':

            q_id, d1_id = self.qrels[index]

            d2_id = np.random.choice(self.doc_ids)
            # making sure that the sampled d1_id is diferent from relevant d2_id
            while d1_id == d2_id:
                d2_id = np.random.choice(self.doc_ids)
        else:
            raise ValueError(f'Unknown split: {split}')

        query = self.queries[q_id]
        doc1 = self.docs[d1_id]
        doc2 = self.docs[d2_id]

        if np.random.random() > 0.5:
            return [query, doc1, doc2], 1
        else:
            return [query, doc2, doc1], -1


class MSMarcoInference(data.Dataset):
    def __init__(self, path, path_ids, debug=False):

        if not debug:
            self.data = read_pickle(path)
            self.ids =  read_pickle(path_ids)
        else:
            self.data = docs_fake
            self.ids = doc_ids_fake

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        return [self.data[id]], id
