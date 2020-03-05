from dataset import MSMarcoSequential
import torch
from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf



# usage: create_index.py model_folder=FOLDER_TO_MODEL
# the script is loading FOLDER_TO_MODEL/best_model.model

def create_index(cfg):
	if not cfg.disable_cuda and torch.cuda.is_available():

		   device = torch.device('cuda')
	else:
		   device = torch.device('cpu')

	# open file
	filename = f'{cfg.dataset_path}/collection.tokenized.tsv'
	ms_batch_generator = MSMarcoSequential(filename, cfg.batch_size).batch_generator()

	# Initialize an Inverted Index object
	ii = InvertedIndex(parent_dir=cfg.model_folder, vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)
	# initialize the index
	print('Initializing index')
	ii.initialize_index()

	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)
	print('Creating index')
	count = 0
	for batch_ids, batch_data, batch_lengths in ms_batch_generator:
		# print(batch_data)
		if count % 1000 == 0:
			print(f'{count} batches processed')
		logits = model(batch_data.to(device), batch_lengths.to(device))
		ii.add_docs_to_index(batch_ids, logits.cpu())
		count += 1

	# save the dictionary with number of latent terms per document in a file
	# ii.save_latent_terms_per_doc_dictionary()
	# sort the posting lists
	ii.sort_posting_lists()

	for doc_id in ii.latent_terms_per_doc:
		print(doc_id, ii.latent_terms_per_doc[doc_id])
	# print(dict ii.)



if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	if not cl_cfg.model_folder:
		raise ValueError("usage: online_inference.py model_folder=FOLDER_TO_MODEL")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	create_index(cfg)
