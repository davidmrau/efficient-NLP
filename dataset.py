import numpy as np
import torch
from torch.utils import data
from utils import *
from torchtext.data.utils import get_tokenizer
from os import path

class MSMarcoTrain(data.Dataset):

    def __init__(self, dataset_path, split):


        self.triplets = read_triplets(f'{dataset_path}/qidpidtriples.{split}.full.tsv')
        self.docs_path = path.join(dataset_path, 'collection/')
        self.queries_path = path.join(dataset_path, f'queries.{split}/')
        self.doc_ids = np.load(f'{self.docs_path}ids.npy')


    def __len__(self):
        # double size, because dataset contain only relevant examples
        return self.triplets.shape[0]

    def __getitem__(self, index):

        q_id, d1_id, d2_id = self.triplets[index]

        query = np.load(f'{self.docs_path}{q_id}.npy')
        doc1 = np.load(f'{self.docs_path}{d1_id}.npy')
        doc2 = np.load(f'{self.docs_path}{d2_id}.npy')

        if np.random.random() > 0.5:
            return [query, doc1, doc2], 1
        else:
            return [query, doc2, doc1], -1


class MSMarcoDev(data.Dataset):

    def __init__(self, dataset_path, split='dev'):


        self.qrels = read_qrels(path.join(dataset_path, f'qrels.{split}.tsv'))
        self.docs_path = path.join(dataset_path, 'collection/')
        self.queries_path = path.join(dataset_path, f'queries.{split}/')
        self.doc_ids = np.load(f'{self.docs_path}ids.npy')


    def __len__(self):
        # double size, because dataset contain only relevant examples
        return self.qrels.shape[0]

    def __getitem__(self, index):

        q_id, d1_id = self.qrels[index]

        d2_id = np.random.choice(self.doc_ids)

        # making sure that the sampled d1_id is diferent from relevant d2_id
        while d1_id == d2_id:
            d2_id = np.random.choice(self.doc_ids)

        query = np.load(f'{self.docs_path}{q_id}.npy')
        doc1 = np.load(f'{self.docs_path}{d1_id}.npy')
        doc2 = np.load(f'{self.docs_path}{d2_id}.npy')

        if np.random.random() > 0.5:
            return [query, doc1, doc2], 1
        else:
            return [query, doc2, doc1], -1
