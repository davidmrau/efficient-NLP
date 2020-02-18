
from utils import get_data_loaders

from snrm import SNRM
dataset_path = ''
train_batch_size = 512
val_batch_size = 512
embedding_dim = 300
hidden_sizes = [512, 512, 512, 512, 10000]
n = 5


pre_trained_embedding_file_name = 'embeddings/glove.6B.300d.txt'

corpus = 'line by line and creates a dictionary mapping words to vectors. For glove.6B.50d.txt this dictionary has 400k words each mapped to a 50 dimensional vector. We can use this to check the values of our pytorch embedding layer. When we use glove to initialize pytorch embedding layers we will only load the words in our corpus vocabulary rather than the full 400k. For my corpus, I only needed 19k vectors.'
vocab = set(corpus.split()) # compute vocab, 6 words
word2idx = {word: idx for idx, word in enumerate(vocab)} # create word index


snrm = SNRM(embedding_dim=embedding_dim, hidden_sizes=hidden_sizes, n=n, pre_trained_embedding_file_name=pre_trained_embedding_file_name, word2idx=word2idx)


dataloaders = get_data_loaders(dataset_path, train_batch_size, val_batch_size)

for x,y, lengths in dataloaders['train']:

    repr = snrm(x, lengths)
