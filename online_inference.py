from dataset import MSMarcoSequential
import torch

from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

""" Load an pre-built inverted index and run online inference (for test set)
"""
# usage: online_inference.py model_folder=FOLDER_TO_MODEL
# the script is loading FOLDER_TO_MODEL/best_model.model
#

# getting command line arguments
cl_cfg = OmegaConf.from_cli()
# getting model config
cfg_load = OmegaConf.load(f'{cl_cfg.model_path}/config.yaml')
# merging both
cfg = OmegaConf.merge(cfg_load, cl_cfg)



def exp():


    if not cfg.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')


    model = torch.load(cfg.model_path)

	# Initialize an Inverted Index object
	ii = InvertedIndex(path=cfg.model_path, vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)

	filename = f'{cfg.dataset_path}/queries.dev.tokenized.tsv'

	ms_batch_generator = MSMarcoSequential(filename, cfg.batch_size).batch_generator()

	model = torch.load(cfg.model_path + '/best_model.model', map_location=device)

	for batch_ids, batch_data, batch_lengths in ms_batch_generator:
		# print(batch_data)
		logits = model(batch_data.to(device), batch_lengths.to(device))
        results = ii.get_scores(ids.cpu().numpy(), logits.cpu())
        print(results)
        exit()


if __name__ == "__main__":
    exp()
