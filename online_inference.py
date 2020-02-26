
from data_loader import get_data_loaders_offline, get_data_loaders_online
import torch
import hydra
from hydra import utils
from inverted_index import InvertedIndex

# from transformers import BertConfig, BertForPreTraining, BertTokenizer


@hydra.main(config_path='config.yaml')

def exp(cfg):

    ii = InvertedIndex(vocab_size = cfg.embedding_dim, num_of_workers=cfg.num_of_workers_index)

    orig_cwd = utils.get_original_cwd() + '/'

    if not cfg.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    dataloaders = get_data_loaders_online(orig_cwd + cfg.dataset_path, cfg.batch_size, debug=cfg.debug)

    model = torch.load(orig_cwd + cfg.model_path)

    for data, ids, lengths in dataloaders['test']:
        logits = model(data.to(device), lengths.to(device))
        results = ii.get_scores(ids, logits.cpu())

if __name__ == "__main__":
    exp()
