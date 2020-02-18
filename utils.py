from dataset import ELI5
from torch.utils.data import DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
import csv


def collate_fn_padd(batch):

    #batch * [q, d1, d2], target

    batch_lengths = list()
    batch_targets = list()
    batch_data = list()
    for item in batch:
        for el in item[0]:
            batch_data.append(el)
            batch_lengths.append(el.shape[0])
        # get sample lengths
        batch_targets.append(item[1])

    batch_targets = torch.stack(batch_targets)

    batch_lengths = torch.LongTensor(batch_lengths)
    #padd data along axis 1
    batch_data = pad_sequence(batch_data,1)

    return [batch_data, batch_targets, batch_lengths]


def get_data_loaders(dataset_path, train_batch_size, val_batch_size):
    dataloaders = {}
    dataloaders['train'] = DataLoader(ELI5(dataset_path+'train.p'), batch_size=train_batch_size, collate_fn=collate_fn_padd)
    dataloaders['test'] =  DataLoader(ELI5(dataset_path+'test.p'), batch_size=val_batch_size, collate_fn=collate_fn_padd)
    dataloaders['val'] =   DataLoader(ELI5(dataset_path+'val.p'), batch_size=val_batch_size, collate_fn=collate_fn_padd)

    return dataloaders


def load_glove_embeddings(path, word2idx, embedding_dim=300):
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float()


def l1_reg_sparse(q_repr, d1_repr, d2_repr):
    return torch.mean(torch.sum(torch.cat([q_repr, d1_repr, d2_repr], dim=1), dim=1))
