import numpy as np
import torch
import torch.nn as nn


from enlp.models.embedding_weighted_average import EmbeddingWeightedAverage

class RankModel(nn.Module):

	def __init__(self, hidden_sizes, embedding_parameters, embedding_dim, vocab_size, dropout_p, weights, trainable_weights, model_type='rank-interaction'):
		super(RankModel, self).__init__()

		self.model_type = model_type

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


		if model_type == 'interaction-based':
			out_size = 1
		elif model_type == 'rank-interaction':
			out_size = 2
		elif model_type == 'representation-based':
			out_size = embedding_dim
		else:
			raise ValueError(f"Model type {model_type} doesn't exist.")
		
		# create module list
		self.layers = nn.ModuleList()
		self.tanh = nn.Tanh()
		if len(hidden_sizes) > 0:
			self.layers.append( nn.Linear(in_features=self.embedding_dim * 2, out_features=hidden_sizes[0]))
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
			self.layers.append( nn.Linear(in_features=embedding_dim * 2, out_features=out_size))
		print(self)

	def forward_interaction(self, q, doc, lengths_q, lengths_d):
		# get embeddings of all inps
		emb_q = self.embedding(q)
		emb_d = self.embedding(doc)
		# calculate weighted average embedding for all inps
		w_av_q = self.weighted_average(q, emb_q, lengths = lengths_q)
		w_av_d = self.weighted_average(doc, emb_d, lengths = lengths_d)
		q_d =  torch.cat([w_av_q, w_av_d], dim=1)
		# getting scores of joint q_d representation
		for layer in self.layers:
			q_d = layer(q_d)
		if self.model_type == 'interaction-based':
			score = self.tanh(q_d)
		score = q_d
		return score.squeeze(1)

	def forward_representation(self, q, lengths_q):
		# get embeddings of all inps
		emb_q = self.embedding(q)
		# calculate weighted average embedding for all inps
		q_d = self.weighted_average(q, emb_q, lengths = lengths_q)
		return q_d
		for layer in self.layers:
			q_d = layer(q_d)
		# getting scores of joint q_d representation
		return q_d

	def forward(self, q, doc, lengths_q=None, lengths_d=None):
		if self.model_type == 'interaction-based':
			return self.forward_interaction(q, doc, lengths_q, lengths_d)
		elif self.model_type == 'representation-based':
			return self.forward_representation(q, doc)


	

