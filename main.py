
from data_loader import get_data_loaders
import torch
from snrm import SNRM
from torch import nn
from run_model import run
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


@hydra.main(config_path='config.yaml')
def exp(cfg):
    print(cfg.pretty())

    orig_cwd = utils.get_original_cwd() + '/'

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
    print('Initializing model...')

    if cfg.model == "snrm":
        # print(cfg.hidden_sizes)
        # print(str2lst(cfg.hidden_sizes))

        model = SNRM(embedding_dim=cfg.snrm.embedding_dim, hidden_sizes=str2lst(cfg.snrm.hidden_sizes),
        sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_path=orig_cwd + embedding_path,
        word2idx=word2idx, dropout_p=cfg.snrm.dropout_p, debug=cfg.debug, device=device)
    elif cfg.model == "tf":
        #model = BERT_based()
        model = BERT_based(  hidden_size = cfg.tf.hidden_size, num_of_layers = cfg.tf.num_of_layers,
        sparse_dimensions = cfg.sparse_dimensions, vocab_size = 30522,
        num_attention_heads = cfg.tf.num_attention_heads, input_length_limit = 150,
        pretrained_embeddings = False, pooling_method = "CLS", device=device)

    if torch.cuda.device_count() > 1:
          print("Let's use", torch.cuda.device_count(), "GPUs!")
              model = nn.DataParallel(model)
    else:
        model = model.to(device=device)

    print(model)
    writer = SummaryWriter(log_dir=f'tb/{datetime.now().strftime("%Y-%m-%d:%H-%M")}/')

    print('Loading data...')
    dataloaders = get_data_loaders(orig_cwd + cfg.dataset_path, cfg.batch_size, debug=cfg.debug)
    print('done')
    loss_fn = nn.MarginRankingLoss().to(device)

    # config_class, model_class, tokenizer_class = BertConfig, BertForMaskedLM, BertTokenizer # MODEL_CLASSES[args.model_type]

    #model = BERT_Based()

    optim = Adam(model.parameters(), lr=cfg.lr)
    print('Start training...')
    model = run(model, dataloaders, optim, loss_fn, cfg.num_epochs, writer, device, l1_scalar=cfg.l1_scalar)

if __name__ == "__main__":
    exp()
