
from data_loader import get_data_loaders
import torch
from snrm import SNRM
from torch import nn
from training import run
from torch.optim import Adam
from bert import BERT_Based
# from transformers import BertConfig, BertForPreTraining, BertTokenizer

import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

train_batch_size = 128
val_batch_size = 128
embedding_dim = 300
hidden_sizes = [300,300, 10000]
n = 5
num_epochs = 20
pre_trained_embedding_file_name = 'embeddings/glove.6B.300d.txt'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word2idx = tokenizer.vocab
dataset_path = 'data/msmarco/'
dataloaders = get_data_loaders(dataset_path, train_batch_size, val_batch_size, tokenizer)

loss_fn = nn.MarginRankingLoss()


# config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer # MODEL_CLASSES[args.model_type]


writer = SummaryWriter()

#model = BERT_Based()

model = SNRM(embedding_dim=embedding_dim, hidden_sizes=hidden_sizes, n=n, pre_trained_embedding_file_name=pre_trained_embedding_file_name, word2idx=word2idx, load_embedding=False)


optim = Adam(model.parameters())

model = run(model, dataloaders, optim, loss_fn, num_epochs, writer)
