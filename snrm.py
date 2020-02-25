

import numpy as np
import torch
import torch.nn as nn
from utils import load_glove_embeddings
import gensim

class SNRM(nn.Module):

    def __init__(self, embedding_dim, hidden_sizes, n, pre_trained_embedding_file_name, word2idx, load_embedding=True):
        super(SNRM, self).__init__()

        self.embedding_dim = embedding_dim
        self.n = n
        self.hidden_sizes = hidden_sizes
        if load_embedding:
            embedding_weights = load_glove_embeddings(pre_trained_embedding_file_name, word2idx, embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        else:
            self.embedding = nn.Embedding(30000, 300)

        self.conv = nn.Conv1d(embedding_dim, hidden_sizes[0], n , stride=1) #input, hidden, filter, stride
        # create module list
        self.linears = nn.ModuleList()
        self.relu = nn.ReLU()
        for k in range(len(hidden_sizes)-1):
            self.linears.append(nn.Conv1d(hidden_sizes[k], hidden_sizes[k+1], 1, stride=1))

    def forward(self, x, lengths):
        # generate mask for averaging over non-zero elements later
        mask = (x > 0)[:, self.n - 1: ]
        out = self.embedding(x)
        out = out.permute(0,2,1)
        out = self.conv(out)  #batch x max_length (n - 1) x hidden

        for i, name in enumerate(self.linears):
            out = self.linears[i](out)
            out= self.relu(out)
        # batch x max_length  - (n-1)x out_size


        mask = mask.unsqueeze(1).repeat(1, self.hidden_sizes[-1],1)
        out = (mask * out).sum(2) / lengths.unsqueeze(1)
        # batch x max_length - (n-1) x out_size
        return out
