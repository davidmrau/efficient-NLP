from dataset import MSMarcoSequential
import torch
from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf



# usage: create_index.py model_folder=FOLDER_TO_MODEL
# the script is loading FOLDER_TO_MODEL/best_model.model


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

	# open file
	filename = f'{cfg.dataset_path}/collection.tokenized.tsv'
	ms_batch_generator = MSMarcoSequential(filename, cfg.batch_size).batch_generator()

	# Initialize an Inverted Index object
	ii = InvertedIndex(path=model_path, vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)
	# initialize the index
	ii.initialize_index()

	model = torch.load(model_path + '/best_model.model', map_location=device)

	for batch_ids, batch_data, batch_lengths in ms_batch_generator:
		# print(batch_data)
		logits = model(batch_data.to(device), batch_lengths.to(device))
		ii.add_docs_to_index(batch_ids, logits.cpu())


	# sort the posting lists
	ii.sort_posting_lists()



if __name__ == "__main__":
	exp()
