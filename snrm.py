

import numpy as np
import torch
import torch.nn as nn
from utils import load_glove_embeddings
import gensim

class SNRM(nn.Module):

    def __init__(self, embedding_dim, hidden_sizes, n, pre_trained_embedding_file_name, word2idx, load_embedding=True):
        super(SNRM, self).__init__()

        self.n = n
        self.embedding_dim = embedding_dim

        if load_embedding:
            embedding_weights = load_glove_embeddings(pre_trained_embedding_file_name, word2idx, embedding_dim)
            self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        else:
            self.embedding = nn.Embedding(1000, embedding_dim)

        self.sliding = nn.Linear(embedding_dim * n, hidden_sizes[0])
        #self.conv = nn.Conv2d(1, self.hidden_size, (5,1), stride=1) #input, hidden, filter, stride
        # create module list
        self.linears = nn.ModuleList()
        self.relu = nn.ReLU()
        for k in range(len(hidden_sizes)-1):
            self.linears.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))


    def forward(self, x, lengths):
        # x (batchsize x longest seq )

        out = self.embedding(x)

        out_list = list()
        # moving window of size n results in num n-grams - length - (n-1) inputs
        for i in range(out.size(1) - (self.n - 1)):
            # get input according to window
            input = out[:, i : i + self.n] # batch_size x num n-grams x embedding_dim
            #reshape input to fit linear n x embedding_dim
            input = input.view(-1, self.n * self.embedding_dim) # batch_size x num n-gram  * embedding_dim
            output = self.sliding(input) # batchsize x num n-grams * hidden_size
            out_list.append(output)

        # stack to tensor
        out = torch.stack(out_list)
        # get n-grams according to length per example
        non_zero_n_grams = [out[:lengths[i] - (self.n - 1),i] for i in range(lengths.size(0))]
        # concatenate non zero n-grams
        out = torch.cat(non_zero_n_grams)

        for i, name in enumerate(self.linears):
            out = self.linears[i](out)
            out= self.relu(out)

        # restore batch_size x n-gram x hidden
        out = [out[sum(lengths[:i]- (self.n - 1)):sum(lengths[:i+1]- (self.n - 1))] for i in range(lengths.size(0))]
        # mean over n-grams
        out = [o.mean(0) for o in out]
        # list to tensor
        out = torch.stack(out)

        return out
