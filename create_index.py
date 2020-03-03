
from data_loader import get_data_loaders_offline
import torch
import hydra
from hydra import utils
from inverted_index import InvertedIndex

# from transformers import BertConfig, BertForPreTraining, BertTokenizer


@hydra.main(config_path='config.yaml')

def exp(cfg):
	# Initialize an Inverted Index object
	ii = InvertedIndex(vocab_size = cfg.embedding_dim, num_of_workers=cfg.num_of_workers_index)
	# initialize the index
	ii.initialize_index()

	orig_cwd = utils.get_original_cwd() + '/'

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	dataloaders = get_data_loaders_offline(orig_cwd + cfg.dataset_path, cfg.batch_size, debug=cfg.debug)
	# load model
	model = torch.load(orig_cwd + cfg.model_path)

	# for each document in the collection, pass it through the model, and use its sparse output vector for indexing
	for data, ids, lengths in dataloaders['docs']:
		logits = model(data.to(device), lengths.to(device))
		ii.add_docs_to_index(ids.cpu().numpy(), logits.cpu())

	# sort the posting lists
	ii.sort_posting_lists()

if __name__ == "__main__":
	exp()
