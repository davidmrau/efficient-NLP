import torch
import torch.nn as nn
import pickle

class EmbeddingWeightedAverage(nn.Module):
	def __init__(self, weights, vocab_size, trainable = True):
		"""
		weights : uniform / random /
				  path_to_file (pickle in the form of tensor.Size(V x 1))
		vocab_size: vocabulary size
		"""
		super(EmbeddingWeightedAverage, self).__init__()

		self.weights = torch.nn.Embedding(num_embeddings = vocab_size, embedding_dim = 1)

		if weights == "uniform":
			self.weights.weight = torch.nn.Parameter(torch.ones(vocab_size,1))
			# pass
		elif weights == "random":
			pass
		# otherwise it has to be a path of a pickle file with the weights in a pytorch tensor form
		else:
			try:
				weight_values = pickle.load(open(weights, 'rb'))
				# print(weight_values.size())
				self.weights.weight = torch.nn.Parameter(weight_values.unsqueeze(-1))
			except:
				raise IOError(f'(EmbeddingWeightedAverage) Loading weights from pickle file: {weights} not accessible!')

		if trainable == False:
			self.weights.weight.requires_grad = False



	def forward(self, inp, values, lengths = None, mask = None):
		"""
		inp shape : Bsz x L
		values shape  : Bsz x L x hidden
		lengths shape : Bsz x 1
		mask: if provided, are of shape Bsx x L. Binary mask version of lenghts
		"""
		if mask is None:

			if lengths is None:
				raise ValueError("EmbeddingWeightedAverage : weighted_average(), mask and lengths cannot be None at the same time!")

			mask = torch.zeros_like(inp)

			for i in range(lengths.size(0)):
				mask[i, : lengths[i].int()] = 1

			if values.is_cuda:
				mask = mask.cuda()

		mask = mask.unsqueeze(-1).float()
		# calculate the weight of each term
		weights = self.weights(inp)

		# normalize the weights
		weights = torch.nn.functional.softmax(weights.masked_fill((1 - mask).bool(), float('-inf')), dim=-1)

		# weights are extended to fit the size of the embeddings / hidden representation
		weights = weights.repeat(1,1,values.size(-1))
		# mask are making sure that we only add the non padded tokens
		mask = mask.repeat(1,1,values.size(-1))
		# we first calculate the weighted sum
		weighted_average = (weights * values * mask).sum(dim = 1)

		return weighted_average


# from torch.optim import Adam



# em = EmbeddingWeightedAverage(weights = "uniform", vocab_size = 10)

# optim = Adam(em.parameters())

# # initialize loss function
# loss_fn = nn.MarginRankingLoss(margin = 0.5)

# optim.zero_grad()



# embeddings = torch.nn.Embedding(num_embeddings = 10, embedding_dim = 5)
# # embeddings.weight = torch.nn.Parameter(torch.ones(10,5) )

# # print(embeddings.weight)
# inp = torch.tensor([[1,3,0], [2,3,5], [6,0,0]])
# lengths = torch.tensor([2,3,1])
# values = embeddings(inp)

# # print(values)

# out = em.weighted_average(inp, values, lengths)

# print(out.size())



# # a = torch.tensor([5,6])
# # b = torch.tensor([1,0]).bool()
# # c = b == 0

# # print(a[b])
# # print(a[c])

# # a[c] == 0




# print(out[0].sum().unsqueeze(0).size())

# print(out[1].sum().unsqueeze(0).size())

# print( torch.tensor([1]).size())


# # calculating loss
# loss = loss_fn(out[0].sum().unsqueeze(0), out[1].sum().unsqueeze(0), torch.tensor([1]))
# # initialize optimizer


# loss.backward()

# optim.step()
