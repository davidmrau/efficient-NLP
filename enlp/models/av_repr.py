import numpy as np
import torch
import torch.nn as nn


from enlp.models.embedding_weighted_average import EmbeddingWeightedAverage

class AvRepr(nn.Module):

	def __init__(self, embedding_parameters, embedding_dim, vocab_size, weights):
		super(AvRepr, self).__init__()

		self.model_type = "representation-based"

		# load or randomly initialize embeddings according to parameters
		if embedding_parameters is None:
			self.embedding_dim = embedding_dim
			self.vocab_size = vocab_size
			self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
			# set embeddings for model
		else:
			self.embedding = nn.Embedding.from_pretrained(embedding_parameters, freeze=False)
			self.embedding_dim = embedding_parameters.size(1)
			self.vocab_size = embedding_parameters.size(0)

		self.weighted_average = EmbeddingWeightedAverage(weights = weights, vocab_size = self.vocab_size, trainable = False) # (weights, vocab_size, trainable = True)
		#self.linear_2 = nn.Linear(embedding_dim, embedding_dim)
		print(self)
	def forward(self, x, lengths):
		# get embeddings of all
		emb = self.embedding(x)
		# calculate weighted average embedding for all inps
		#out = self.weighted_average(x, emb, lengths = lengths)
		out = emb.mean(1)
		#out = emb.mean(1)
		#out = self.linear_2(out
		return out
