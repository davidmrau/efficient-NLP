
from utils import get_data_loaders
import torch
from snrm import SNRM
from torch import nn
from training import train
from torch.optim import Adam
from bert import BERT_Based
# from transformers import BertConfig, BertForPreTraining, BertTokenizer

import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

dataset_path = 'toy_dataset'
train_batch_size = 8
val_batch_size = 8
embedding_dim = 300
hidden_sizes = [128, 32, 128]
n = 5
num_epochs = 10
pre_trained_embedding_file_name = 'embeddings/glove.6B.300d.txt'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
word2idx = tokenizer.vocab
dataloaders = get_data_loaders(dataset_path, train_batch_size, val_batch_size, tokenizer)

loss_fn = nn.MarginRankingLoss()


# config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer # MODEL_CLASSES[args.model_type]


writer = SummaryWriter()

#model = BERT_Based()

model = SNRM(embedding_dim=embedding_dim, hidden_sizes=hidden_sizes, n=n, pre_trained_embedding_file_name=pre_trained_embedding_file_name, word2idx=word2idx, load_embedding=True)


optim = Adam(model.parameters())

model = train(model, dataloaders['train'], optim, loss_fn, num_epochs, writer)
