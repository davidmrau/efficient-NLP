import pickle
import torch



word2id = pickle.load(open("data/embeddings/glove.6B.300d_word2idx_dict.p", 'rb'))

idf_dict = pickle.load(open("data/robust04/robust04_raw_docs_idf_dict.pickle", 'rb'))

average_idf = 0

token_id_2_idf = {}

for word in idf_dict:

	word_id = word2id[word]

	token_id_2_idf[word_id] = idf_dict[word]

	average_idf += idf_dict[word]

average_idf /= len(idf_dict)


vocab_size = len(word2id)
idf_tensor = torch.ones(vocab_size)

for i in range(vocab_size):

	if i in token_id_2_idf:
		idf_tensor[i] = token_id_2_idf[i]
	else:
		idf_tensor[i] = average_idf

# set idf of "[PAD]" token to 0!
idf_tensor[0] = 0

pickle.dump(idf_tensor, open("data/embeddings/idf_tensor.p","wb"))