import numpy as np

import matplotlib.pyplot as plt

def get_posting_lengths(reprs, sparse_dims):
	# lengths = np.zeros(sparse_dims)
	return lengths


def plot_ordered_posting_lists_lengths(path=".",reprs = None, name = "", n=-1):
	sparse_dims = reprs.shape[1]

	frequencies = (reprs !=0).sum(0)

	n = n if n > 0 else len(frequencies)
	top_n = sorted(frequencies, reverse=True)[:n]

	# run matplotlib on background, not showing the plot

	plt.plot(top_n)
	plt.ylabel('Frequency')

	n_text = f' (top {n})' if n != len(frequencies) else ''
	plt.xlabel('Latent Dimension (Sorted)' + n_text)
	plt.show()
	plt.savefig(path+ f'/num_docs_per_term.pdf', bbox_inches='tight')
	plt.close()


filename = "/project/draugpu/data/robust04/robust04_raw_docs.num_query_glove_stop_lucene_remove_unk_max_len_1500.tsv"

reprs = []

# temp_numpy = np.zeros(400002, dtype=bool)

# print(temp_numpy.shape)

# temp_numpy[0] = 5

# print(np.sum(temp_numpy))
# exit()





file = open(filename, "r")

for line in file:

	# print(line)
	temp = line.split("\t")

	id = temp[0]
	content = temp[1]

	token_ids = content.split()

	temp_numpy = np.zeros(400002, dtype=bool)

	for token_id in token_ids:
		temp_numpy[ int(token_id) ] += 1
		# token_id = int(token_id)

	# token_ids = [int(token_id) for token_id in token_ids]




	reprs.append(temp_numpy)

	# print(id)
	# print(content)
	# print(token_ids)
	# exit()
	# break

reprs = np.stack(reprs, axis = 0)

# print(reprs.shape)
# exit()


plot_ordered_posting_lists_lengths(reprs =reprs)


# import pickle
# import torch



# word2id = pickle.load(open("data/embeddings/glove.6B.300d_word2idx_dict.p", 'rb'))

# idf_dict = pickle.load(open("data/robust04/robust04_raw_docs_idf_dict.pickle", 'rb'))

# average_idf = 0

# token_id_2_idf = {}

# for word in idf_dict:

# 	word_id = word2id[word]

# 	token_id_2_idf[word_id] = idf_dict[word]

# 	average_idf += idf_dict[word]

# average_idf /= len(idf_dict)


# vocab_size = len(word2id)
# idf_tensor = torch.ones(vocab_size)

# for i in range(vocab_size):

# 	if i in token_id_2_idf:
# 		idf_tensor[i] = token_id_2_idf[i]
# 	else:
# 		idf_tensor[i] = average_idf

# # set idf of "[PAD]" token to 0!
# idf_tensor[0] = 0

# pickle.dump(idf_tensor, open("data/embeddings/idf_tensor.p","wb"))