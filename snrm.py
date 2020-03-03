

import numpy as np
import torch
import torch.nn as nn
from utils import load_glove_embeddings, get_pretrained_BERT_embeddings
import gensim

class SNRM(nn.Module):

    def __init__(self, hidden_sizes, sparse_dimensions, n, embedding_path, word2idx, dropout_p, device, debug=False):
        super(SNRM, self).__init__()

        self.embedding_dim = 300
        self.n = n
        self.hidden_sizes = hidden_sizes
        if embedding_path != '':
            # load the appropriate embeddings
            if embedding_path == 'bert':
                embedding_weights = get_pretrained_BERT_embeddings()
                self.embedding_dim = embedding_weights.size(1)
            else:
                embedding_weights = load_glove_embeddings(embedding_path, word2idx, device)
                self.embedding_dim = 300
                # set embeddings for model
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        else:
            self.embedding = nn.Embedding(30000, self.embedding_dim)
        self.sparse_dimensions = sparse_dimensions
        self.conv = nn.Conv1d(self.embedding_dim, hidden_sizes[0], n , stride=1) #input, hidden, filter, stride
        # create module list
        self.linears = nn.ModuleList()
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout_p)
        for k in range(len(hidden_sizes)-1):
            self.linears.append(nn.Conv1d(hidden_sizes[k], hidden_sizes[k+1], 1, stride=1))

        self.linears.append(nn.Conv1d(hidden_sizes[-1], sparse_dimensions, 1, stride=1))

    def forward(self, x, lengths):
        # generate mask for averaging over non-zero elements later
        mask = (x > 0)[:, self.n - 1: ]
        out = self.embedding(x)

        out = out.permute(0,2,1)
        out = self.conv(out)  #batch x max_length (n - 1) x hidden

        out= self.relu(out)
        out = self.drop(out)

        for i in range(len(self.linears)-1):
            out = self.linears[i](out)
            out= self.relu(out)
            out = self.drop(out)

        # we do not apply dropout on the last layer
        out = self.linears[-1](out)
        out= self.relu(out)


        # batch x max_length  - (n-1)x out_size


        mask = mask.unsqueeze(1).repeat(1, self.sparse_dimensions,1).float()
        out = (mask * out).sum(2) / lengths.unsqueeze(1)
        # batch x max_length - (n-1) x out_size
        return out

    def get_optimizer(self):
        return Adam(self.parameters(), lr=cfg.lr)
