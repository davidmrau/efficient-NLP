from inverted_index import InvertedIndex
from omegaconf import OmegaConf
import numpy as np
import torch
# usage: create_index.py model_folder=FOLDER_TO_MODEL
# the script is loading FOLDER_TO_MODEL/best_model.model

def create_index(cfg):



	count = 0
	with open(cfg.docs_file, 'r') as f:
		ids, logits = list(), list()
		for line in f:
		# print(batch_data)
			
			delim_pos = line.find('\t')
			# extracting the id
			id_ = line[:delim_pos]
			# extracting the token_ids and creating a numpy array
			logit = np.fromstring(line[delim_pos+1:], dtype=float, sep=' ')

			if count == 0:

				# Initialize an Inverted Index object
				ii = InvertedIndex(parent_dir=cfg.out, vocab_size = len(logit), num_of_decimals=cfg.decimals)
				# initialize the index
				print('Initializing index')
				ii.initialize_index()
				print('Creating index')

			ids.append(id_)
			logits.append(logit)
		
			if count % 10000 == 0:
				print(f'{count} documents read, processing now')
				ii.add_docs_to_index(ids, torch.Tensor(logits))
				ids, logits = list(), list()
			count += 1

	# save the dictionary with number of latent terms per document in a file
	ii.save_latent_terms_per_doc_dictionary()
	# sort the posting lists
	ii.sort_posting_lists()

	ii.get_and_save_dict_of_lenghts_per_posting_list()
	length_of_posting_lists_dict = ii.read_posting_lists_lengths_dictionary()
	latent_lengths_dict = ii.load_latent_terms_per_doc_dictionary()
	ii.plot_histogram_of_latent_terms(list(latent_lengths_dict.values()))
	ii.plot_ordered_posting_lists_lengths(list(length_of_posting_lists_dict.values()), n=1000)

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	if not cl_cfg.docs_file or not cl_cfg.out or not cl_cfg.decimals:
		raise ValueError("usage: create_index.py out=INDEX_PATH docs_file=PATH_TO_DOCS_FILE decimals=5")
	# getting model config
	create_index(cl_cfg)
