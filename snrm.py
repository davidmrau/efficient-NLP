

import numpy as np
import torch
import torch.nn as nn
from utils import load_glove_embeddings
import gensim

class SNRM(nn.Module):

    def __init__(self, embedding_dim, hidden_sizes, n, pre_trained_embedding_file_name, word2idx):
        super(SNRM, self).__init__()

        self.n = n
        embedding_weights = load_glove_embeddings(pre_trained_embedding_file_name, word2idx, embedding_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding_weights, freeze=False)
        self.sliding = nn.Linear(embedding_dim * n, hidden_sizes[0])
        #self.conv = nn.Conv2d(1, self.hidden_size, (5,1), stride=1) #input, hidden, filter, stride
        # create module list
        self.linears = nn.ModuleList()
        self.relu = nn.ReLU()
        for k in range(len(hidden_sizes)-1):
            self.linears.append(nn.Linear(hidden_sizes[k], hidden_sizes[k+1]))


    def forward(self, x, lengths):
        # x (batchsize x longest seq x vocab_size)

        out = self.embedding(x)

        out_list = list()
        # moving window of size n results in num n-grams - length - (n-1) inputs
        for i in range(out.size(1) - (self.n - 1)):
            # get input according to window
            input = out[:, i : i + self.n]
            #reshape input to fit linear n * embedding_dim
            input = input.view(-1, self.n * out.size(2))
            output = self.sliding(input)
            out_list.append(output)

        # stack to tensor
        out = torch.stack(out_list)
        # get n-grams according to length per example
        non_zero_n_grams = [out[:lengths[i],i] for i in range(lengths.size(0))]
        # concatenate non zero n-grams
        out = torch.cat(non_zero_n_grams)


        for i, name in enumerate(self.linears):
            out = self.linears[i](out)
            out= self.relu(out)

        # restore batch_size x n-gram x hidden
        out = [out[sum(lengths[:i]):sum(lengths[:i])+lengths[i]] for i in range(lengths.size(0))]
        # mean over n-grams
        out = [o.mean(0) for o in out]
        # list to tensor
        out = torch.stack(out)
        return out
