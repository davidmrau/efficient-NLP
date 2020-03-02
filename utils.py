
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from os import path
import json
import pickle
import csv
import subprocess
import transformers


def file_len(fname):
    """ Get the number of lines from file
    """ # https://stackoverflow.com/questions/845058/how-to-get-line-count-of-a-large-file-cheaply-in-python
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])



def collate_fn_padd(batch):
    """ Collate function for aggregating samples into batch size.
        returns:
        batch_data = Torch([ q_1, q_2, ..., q1_d1, q2_d1, ..., q2_d1, q2_d2, ... ])
        batch_targets = -1 or 1 for each sample
        batch_lenghts = lenght of each query/document,
            that is used for proper averaging ignoring 0 padded inputs
    """

    #batch * [q, d1, d2], target

    batch_lengths = list()
    batch_targets = list()
    batch_q, batch_doc1, batch_doc2 = list(), list(), list()
    for item in batch:
        q, doc1, doc2 = item[0]
        batch_q.append(torch.ShortTensor(q))
        batch_doc1.append(torch.ShortTensor(doc1))
        batch_doc2.append(torch.ShortTensor(doc2))
        batch_targets.append(item[1])
    batch_data = batch_q + batch_doc1 + batch_doc2
    batch_targets = torch.Tensor(batch_targets)

    batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
    #padd data along axis 1
    batch_data = pad_sequence(batch_data,1).long()
    return batch_data, batch_targets, batch_lengths



def get_pretrained_BERT_embeddings():
    bert = transformers.BertModel.from_pretrained('bert-base-uncased')
    return bert.embeddings.word_embeddings.weight



def load_glove_embeddings(path, word2idx, device, embedding_dim=300):
    """ Load Glove embeddings from file
    """
    with open(path) as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word2idx.get(word)
            if index:
                vector = np.array(values[1:], dtype='float32')
                embeddings[index] = vector
        return torch.from_numpy(embeddings).float().to(device)


def l1_loss(q_repr, d1_repr, d2_repr):
    """ L1 loss ( Sum of vectors )
    """
    concat = torch.cat([q_repr, d1_repr, d2_repr], 1)
    return torch.mean(torch.sum(concat, 1))/q_repr.size(1)


def l0_loss(q_repr, d1_repr, d2_repr):
    """ L0 loss ( Number of non zero elements )
    """
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
    with open(path, newline='') as csvfile:
        return list(csv.reader(csvfile, delimiter=delimiter))

def read_qrels(path, delimiter='\t'):
    data = list()
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line_split = line.strip().split(delimiter)
            data.append([int(line_split[0]), int(line_split[2])])
    return data

def read_pickle(path):
    return pickle.load(open(path, 'rb'))

def read_json(path):
    with open(path, "r") as read_file:
        return json.load(read_file)


def str2lst(string):
    return [int(s) for s in string.split('-')]

def create_seek_dictionary_per_index(filename):
    """ Creating a dictionary, for accessing directly a documents content, given document's id
        from a large file containing all documents
        returns:
        dictionary [doc_id] -> Seek value of a large file, so that you only have to read the exact document (doc_id)
    """
    index_to_seek = list()
    sample_counter = 0

    with open(filename) as file:
        seek_value = file.tell()
        index_to_seek.append(seek_value)
        line = file.readline()
        while(line != ""):
            sample_counter += 1
            seek_value = file.tell()
            index_to_seek.append(seek_value)
            line = file.readline()
            if sample_counter % 100000 == 0:
                print(sample_counter)

    del index_to_seek[ -1 ]

    return index_to_seek

def get_index_line_from_file(file, index_seek_dict, index):
    """ Given a seek value and a file, read the line that follows that seek value
    """
    file.seek( index_seek_dict[index] )
    return file.readline()


def get_ids_from_tsv(line):
    delim_pos = line.find(' ')
    id_ = int(line[:delim_pos])
    ids = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
    return id_, ids
