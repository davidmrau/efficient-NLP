import numpy as np
import torch
from inverted_index import InvertedIndex


def get_sparse_representation(num_docs, dim=1000, sparsity_ratio=0.9):
    rand = np.random.uniform(0, 1, size=(num_docs, dim))
    rand[rand <= sparsity_ratio] = 0
    return torch.Tensor(rand)

# example usage

def create_index():
	
	num_of_decimals = 5
	latent_terms = 100
	num_docs = 200
	index_dir = 'example_index'

	ii = InvertedIndex(parent_dir=index_dir, vocab_size = latent_terms, num_of_decimals=num_of_decimals)
	# initialize index
	ii.initialize_index()
	# exemplary input
	sparse_representation = get_sparse_representation(num_docs, dim=latent_terms)
	doc_ids = [str(i) for i in range(sparse_representation.size(0))] 
	# add to index
	ii.add_docs_to_index(doc_ids, sparse_representation)
	# save the dictionary with number of latent terms per document in a file
	ii.save_latent_terms_per_doc_dictionary()
	# sort the posting lists
	ii.sort_posting_lists()


if __name__ == "__main__":
	# getting command line arguments
	create_index()
