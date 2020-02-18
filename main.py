
from utils import get_data_loaders
import torch
from snrm import SNRM
from torch import nn
from training import train
from torch.optim import Adam



dataset_path = ''
train_batch_size = 8
val_batch_size = 8
embedding_dim = 300
hidden_sizes = [128, 32, 128]
n = 5
num_epochs = 10
pre_trained_embedding_file_name = 'embeddings/glove.6B.300d.txt'



corpus = 'line by line and creates a dictionary mapping words to vectors. For glove.6B.50d.txt this dictionary has 400k words each mapped to a 50 dimensional vector. We can use this to check the values of our pytorch embedding layer. When we use glove to initialize pytorch embedding layers we will only load the words in our corpus vocabulary rather than the full 400k. For my corpus, I only needed 19k vectors.'
vocab = set(corpus.split()) # compute vocab, 6 words
word2idx = {word: idx for idx, word in enumerate(vocab)} # create word index





dataloaders = get_data_loaders(dataset_path, train_batch_size, val_batch_size)

loss_fn = nn.MarginRankingLoss()

snrm = SNRM(embedding_dim=embedding_dim, hidden_sizes=hidden_sizes, n=n, pre_trained_embedding_file_name=pre_trained_embedding_file_name, word2idx=word2idx, load_embedding=False)

optim = Adam(snrm.parameters())

model, loss_logs_train = train(snrm, dataloaders['train'], optim, loss_fn, num_epochs)
