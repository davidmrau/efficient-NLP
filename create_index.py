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
	ms_batch_generator = MSMarcoSequential(cfg.dataset_path + cfg.docs_file, cfg.batch_size).batch_generator()

	# Initialize an Inverted Index object
	ii = InvertedIndex(parent_dir=cfg.model_folder, vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)
	# initialize the index
	print('Initializing index')
	ii.initialize_index()

	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)
	print('Creating index')

	# nom_docs = 1000
	#
	# temp = torch.rand(nom_docs, cfg.sparse_dimensions)
	# mask = torch.randn(nom_docs, cfg.sparse_dimensions) > 0.8
	# temp = temp* mask.float()
	# doc_ids = [str(i) for i in range(nom_docs)]
	# # temp = temp* (temp > 0)
	# ii.add_docs_to_index(doc_ids, temp.cpu())
	#
	# ii.save_latent_terms_per_doc_dictionary()
	#
	# ii.sort_posting_lists()
	#
	# ii.get_and_save_dict_of_lenghts_per_posting_list()
	#
	# length_of_posting_lists_dict = ii.read_posting_lists_lengths_dictionary()
	#
	# latent_lengths_dict = ii.load_latent_terms_per_doc_dictionary()
	#
	# ii.plot_histogram_of_latent_terms(list(latent_lengths_dict.values()))
	#
	# ii.plot_ordered_posting_lists_lengths(list(length_of_posting_lists_dict.values()), n=-1)
	#
	# exit()
	#

	count = 0
	for batch_ids, batch_data, batch_lengths in ms_batch_generator:
		# print(batch_data)
		if count % 1000 == 0:
			print(f'{count} batches processed')
		logits = model(batch_data.to(device), batch_lengths.to(device))
		ii.add_docs_to_index(batch_ids, logits.cpu())
		count += 1

	# save the dictionary with number of latent terms per document in a file
	ii.save_latent_terms_per_doc_dictionary()
	# sort the posting lists
	ii.sort_posting_lists()



if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	if not cl_cfg.model_folder or not cl_cfg.docs_file :
		raise ValueError("usage: create_index.py model_folder=MODEL_FOLDER docs_file=PATH_TO_DOCS_FILE (in dataset_path)")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	create_index(cfg)
