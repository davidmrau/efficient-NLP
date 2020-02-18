


import numpy as np

import torch
from torch.utils import data

class ELI5(data.Dataset):

  def __init__(self, dataset_path):
      pass

  def __len__(self):
        return 100000

  def __getitem__(self, index, vocab_size= 20):

        # 'Generates one sample of data'
        # # Select sample
        # ID = self.list_IDs[index]
        #
        # # Load data and get label
        # X = torch.load('data/' + ID + '.pt')
        # y = self.labels[ID]
        #

        X = torch.zeros(np.random.randint(20)).long().random_(1, 20)
        y = 0

        return X, y
