import numpy as np
import torch
import torch.nn as nn


from enlp.models.embedding_weighted_average import EmbeddingWeightedAverage

class RankProbModel(nn.Module):

	def __init__(self, hidden_sizes, embedding_parameters, embedding_dim, vocab_size, dropout_p, weights, trainable_weights):
		super(RankProbModel, self).__init__()

		self.model_type = "rank_prob"

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

		out_size = 2

		# create module list
		self.layers = nn.ModuleList()
		self.sigmoid = nn.Sigmoid()
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=self.embedding_dim * 3, out_features=hidden_sizes[0]))
			#self.layers.append( nn.Linear(in_features=self.embedding_dim, out_features=hidden_sizes[0]))
			self.layers.append(nn.Dropout(p=dropout_p))
			self.layers.append(nn.ReLU())

		for k in range(len(hidden_sizes)-1):
			self.layers.append(nn.Linear(in_features=hidden_sizes[k], out_features=hidden_sizes[k+1]))
			self.layers.append(nn.Dropout(p=dropout_p))
			self.layers.append(nn.ReLU())

		#self.linear = nn.Linear(in_features=self.embedding_dim, out_features=self.embedding_dim)
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=hidden_sizes[-1], out_features=out_size))
		else:
			self.layers.append( nn.Linear(in_features=embedding_dim * 3, out_features=out_size))

		print(self)

	def get_av_repr(self, x, lengths_x):
		emb_x = self.embedding(x)
		av = self.weighted_average(x, emb_x, lengths = lengths_x)
		return av

	def forward(self, q, doc1, doc2=None, lengths_q=None, lengths_d1=None, lengths_d2=None, av_provided=False, get_av_repr=False):
		if get_av_repr:
			return self.get_av_repr(q, doc1)
		if not av_provided:
			# get embeddings of all inps
			emb_q = self.embedding(q)
			emb_d1 = self.embedding(doc1)
			emb_d2 = self.embedding(doc2)
			# calculate weighted average embedding for all inps
			w_av_q = self.weighted_average(q, emb_q, lengths = lengths_q)
			w_av_d1 = self.weighted_average(doc1, emb_d1, lengths = lengths_d1)
			w_av_d2 = self.weighted_average(doc2, emb_d2, lengths = lengths_d2)
			q_d =  torch.cat([w_av_q, w_av_d1, w_av_d2], dim=1)
		else:
			b_size = doc2.shape[0]
			q = q.repeat(b_size, 1)
			doc1 = doc1.repeat(b_size, 1)
			q_d =  torch.cat([q, doc1, doc2], dim=1)
		# getting scores of joint q_d representation
		for layer in self.layers:
			q_d = layer(q_d)
		score = self.sigmoid(q_d)
		score = q_d
		return score.squeeze(1)
