import numpy as np
import torch
import torch.nn as nn


from enlp.models.embedding_weighted_average import EmbeddingWeightedAverage

class RankModel(nn.Module):

	def __init__(self, hidden_sizes, embedding_parameters, embedding_dim, vocab_size, dropout_p, weights, trainable_weights):
		super(RankModel, self).__init__()

		self.model_type = "interaction-based"

		self.hidden_sizes = hidden_sizes

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
		
		self.weighted_average = EmbeddingWeightedAverage(weights = weights, vocab_size = self.vocab_size, trainable = trainable_weights) # (weights, vocab_size, trainable = True)
		print(weights)
		# create module list
		self.linears = nn.ModuleList()
		self.relu = nn.ReLU()
		self.drop = nn.Dropout(p=dropout_p)
		self.tanh = nn.Tanh()

		self.linears.append( nn.Linear(in_features=self.embedding_dim * 2, out_features=hidden_sizes[0]))

		for k in range(len(hidden_sizes)-1):
			self.linears.append( nn.Linear(in_features=hidden_sizes[k], out_features=hidden_sizes[k+1]))

		self.linears.append( nn.Linear(in_features=hidden_sizes[-1], out_features=1))


	def forward(self, q, doc, lengths_q, lengths_d):

		inp = torch.cat([q, doc])
		lengths = torch.cat([lengths_q, lengths_d])
		# get embeddings of all inps
		out = self.embedding(inp)
		print(inp.max())
		# calculate weighted average embedding for all inps
		weight_averaged = self.weighted_average(inp, out,  lengths = lengths)
		
		split_size = weight_averaged.size(0) // 2
		q, d = torch.split(weight_averaged, split_size)
		q_d =  torch.cat([q, d], dim=1)

		# getting scores of joint q_d representation
		for i in range(len(self.linears)-1):
			q_d = self.linears[i](q_d)
			q_d = self.relu(q_d)
			q_d = self.drop(q_d)

		# we do not apply dropout on the last layer
		q_d = self.linears[-1](q_d)

		score = self.tanh(q_d)

		return score
