
from data_loader import get_data_loaders
import torch
from snrm import SNRM
from torch import nn
from training import run
from torch.optim import Adam
from bert import BERT_Based
import hydra
from hydra import utils
import os
from datetime import datetime

# from transformers import BertConfig, BertForPreTraining, BertTokenizer
from utils import str2lst
import transformers
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer

@hydra.main(config_path='config.yaml')
def exp(cfg):


    if not cfg.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if cfg.embedding == 'glove':
        embedding_path = cfg.glove_embedding_path
    elif cfg.embedding == 'bert':
        embedding_path = cfg.bert_embedding_path


    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    word2idx = tokenizer.vocab
    dataloaders = get_data_loaders(cfg.dataset_path, cfg.batch_size, tokenizer, debug=cfg.debug)

    loss_fn = nn.MarginRankingLoss()

    # config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer # MODEL_CLASSES[args.model_type]
    writer = SummaryWriter(log_dir=f'tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')

    #model = BERT_Based()

    model = SNRM(embedding_dim=cfg.embedding_dim, hidden_sizes=str2lst(cfg.hidden_sizes),
    n=cfg.n, embedding_path=embedding_path,
    word2idx=word2idx, dropout_p=cfg.dropout_p, debug=cfg.debug).to(device=args.device)

    optim = Adam(model.parameters())

    model = run(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device, l1_scalar=cfg.l1_scalar)

if __name__ == "__main__":
    exp()
