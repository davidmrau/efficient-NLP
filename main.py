
from data_loader import get_data_loaders
import torch
from snrm import SNRM
from torch import nn
from run_model import train
from torch.optim import Adam
from bert_based import BERT_based
import hydra
from hydra import utils
import os
from datetime import datetime

# from transformers import BertConfig, BertForPreTraining, BertTokenizer
from utils import str2lst
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

# loading 'params' using facebook's hydra
@hydra.main(config_path='config.yaml')
def exp(cfg):
    # printing params
    print(cfg.pretty())

    orig_cwd = utils.get_original_cwd() + '/'
    # select device depending on availability and user's setting
    if not cfg.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # define which embeddings to load, depending on params
    if cfg.embedding == 'glove':
        embedding_path = cfg.glove_embedding_path
    elif cfg.embedding == 'bert':
        embedding_path = cfg.bert_embedding_path

    # load BERT's BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # use BERT's word to ID
    word2idx = tokenizer.vocab
    print('Initializing model...')

    # initialize model according to params (SNRM or BERT-like Transformer Encoder)
    if cfg.model == "snrm":
        model = SNRM(embedding_dim=cfg.snrm.embedding_dim, hidden_sizes=str2lst(cfg.snrm.hidden_sizes),
        sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_path=orig_cwd + embedding_path,
        word2idx=word2idx, dropout_p=cfg.snrm.dropout_p, debug=cfg.debug, device=device)

    elif cfg.model == "tf":
        model = BERT_based(  hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
        sparse_dimensions = cfg.sparse_dimensions, vocab_size = 30522,
        num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = 150,
        pretrained_embeddings = False, pooling_method = "CLS", device=device)

    # move model to device
    model = model.to(device=device)

    print(model)
    # initialize tensorboard
    writer = SummaryWriter(log_dir=f'tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')

    print('Loading data...')
    # initialize dataloaders
    dataloaders = get_data_loaders(orig_cwd + cfg.dataset_path, cfg.batch_size, debug=cfg.debug)
    print('done')
    # initialize loss function
    loss_fn = nn.MarginRankingLoss(margin = 1.0).to(device)


    # initialize optimizer
    optim = Adam(model.parameters(), lr=cfg.lr)
    print('Start training...')
    # train the model
    model = train(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device, l1_scalar=cfg.l1_scalar)

if __name__ == "__main__":
    exp()
