


import numpy as np

import torch
from torch.utils import data

class ELI5(data.Dataset):

  def __init__(self, dataset_path):
      pass

  def __len__(self):
        return 100

  def __getitem__(self, index, vocab_size= 20):

        q = torch.zeros(np.random.randint(6,40)).long().random_(1, 40)
        d1 = torch.zeros(np.random.randint(6,40)).long().random_(1, 40)
        d2 = torch.zeros(np.random.randint(6,40)).long().random_(1, 40)
        target = torch.LongTensor([1 if np.random.rand() > 0.5 else -1])
        return [q, d1, d2], target
