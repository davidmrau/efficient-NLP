import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path
from fake_data import *
class MSMarcoTrain(data.Dataset):

    def __init__(self, dataset_path, split='train', debug_mode=False):


        if not debug_mode:
            self.triplets = read_triplets(f'{dataset_path}/qidpidtriples.{split}.full.tsv')
            self.docs = read_pickle(f'{dataset_path}/collection.tsv.p')
            self.queries = read_pickle(f'{dataset_path}/queries.{split}.tsv.p')
        else:
            self.triplets = triplets_fake
            self.docs = docs_fake
            self.queries = queries_fake

    def __len__(self):
        # double size, because dataset contain only relevant examples
        return self.triplets.shape[0]

    def __getitem__(self, index):

        q_id, d1_id, d2_id = self.triplets[index]

        query = self.queries[q_id]
        doc1 = self.docs[d1_id]
        doc2 = self.docs[d2_id]

        if np.random.random() > 0.5:
            return [query, doc1, doc2], 1
        else:
            return [query, doc2, doc1], -1


class MSMarcoDev(data.Dataset):

    def __init__(self, dataset_path, split='dev', debug_mode=False):

        if not debug_mode:
            self.qrels = read_qrels(path.join(dataset_path, f'qrels.{split}.tsv'))
            self.docs = read_pickle(f'{dataset_path}/collection.tsv.p')
            self.queries = read_pickle(f'{dataset_path}/queries.{split}.tsv.p')
            self.doc_ids = np.load(f'{dataset_path}/doc_ids.npy')
        else:
            self.qrels = qrels_fake 
            self.triplets = triplets_fake
            self.docs = docs_fake
            self.queries = queries_fake
            self.doc_ids = doc_ids_fake


    def __len__(self):
        # double size, because dataset contain only relevant examples
        return len(self.qrels)

    def __getitem__(self, index):

        q_id, d1_id = self.qrels[index]

        d2_id = np.random.choice(self.doc_ids)
        # making sure that the sampled d1_id is diferent from relevant d2_id
        while d1_id == d2_id:
            d2_id = np.random.choice(self.doc_ids)


        query = self.queries[q_id]
        doc1 = self.docs[d1_id]
        doc2 = self.docs[d2_id]

        if np.random.random() > 0.5:
            return [query, doc1, doc2], 1
        else:
            return [query, doc2, doc1], -1
