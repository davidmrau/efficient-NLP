
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from os import path
import json
import pickle


def collate_fn_padd(batch):

    #batch * [q, d1, d2], target

    batch_lengths = list()
    batch_targets = list()
    batch_data = list()
    for item in batch:
        for el in item[0]:
            batch_data.append(torch.LongTensor(el))
            batch_lengths.append(len(el))
        # get sample lengths
        batch_targets.append(item[1])

    batch_targets = torch.Tensor(batch_targets)

    batch_lengths = torch.LongTensor(batch_lengths)
    #padd data along axis 1
    batch_data = pad_sequence(batch_data,1).long()


    return batch_data, batch_targets, batch_lengths




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


def l1_loss(q_repr, d1_repr, d2_repr):
    concat = torch.cat([q_repr, d1_repr, d2_repr], 1)
    return torch.mean(torch.sum(concat, 1))/q_repr.size(1)


def l0_loss(q_repr, d1_repr, d2_repr):
    # return mean batch l0 loss of qery, and docs
    concat_d = torch.cat([d1_repr, d2_repr], dim=1)
    non_zero_d = (concat_d > 0).float().mean(1).mean()
    non_zero_q = (q_repr > 0).float().mean(1).mean()
    return non_zero_d, non_zero_q



def read_csv(path, delimiter='\t'):
    return np.genfromtxt(path, delimiter=delimiter)

def read_data(path, delimiter='\t'):
    with open(path, 'r') as f:
        ids = list()
        data = list()
        for line in f.readlines():
            line_split = line.split(delimiter, 1)
            ids.append(line_split[0])
            data.append(line_split[1])
        return np.asarray(ids), np.asarray(data)


def read_triplets(path, delimiter='\t'):
    with open(path, 'r') as f:
        data = list()
        for line in f.readlines():
            line_split = line.strip().split(delimiter)
            data.append(line_split)
        return np.asarray(data)

def read_qrels(path, delimiter='\t'):
    data = list()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_split = line.strip().split(delimiter)
            data.append([line_split[0], line_split[2]])
    return data

def read_pickle(path):
    return pickle.load(open(path, 'rb'))

def read_json(path):
    with open(path, "r") as read_file:
        return json.load(read_file)
