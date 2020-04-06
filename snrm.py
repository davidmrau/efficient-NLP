

import numpy as np
import torch
import torch.nn as nn
from utils import load_glove_embeddings, get_pretrained_BERT_embeddings
import gensim
import transformers

class SNRM(nn.Module):

	def __init__(self, hidden_sizes, sparse_dimensions, n, embedding_parameters, embedding_dim, vocab_size, dropout_p, n_gram_model = 'cnn', large_out_biases = False):
		super(SNRM, self).__init__()

		self.n = n
		self.hidden_sizes = hidden_sizes
		self.sparse_dimensions = sparse_dimensions
		self.n_gram_model = n_gram_model

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


		if n_gram_model == 'cnn':
			self.conv = nn.Conv1d(self.embedding_dim, hidden_sizes[0], n , stride=1) #input, hidden, filter, stride
		elif n_gram_model == 'bert':
			# initialize bert
			self.init_transformer()


		# create module list
		self.linears = nn.ModuleList()
		self.relu = nn.ReLU()
		self.drop = nn.Dropout(p=dropout_p)
		for k in range(len(hidden_sizes)-1):
			self.linears.append(nn.Conv1d(hidden_sizes[k], hidden_sizes[k+1], 1, stride=1))

		self.linears.append(nn.Conv1d(hidden_sizes[-1], sparse_dimensions, 1, stride=1))

		if large_out_biases:
			self.linears[-1].bias = torch.nn.Parameter(torch.ones(sparse_dimensions) * 3 )

	def forward(self, x, lengths):
		# generate mask for averaging over non-zero elements later
		mask = (x > 0)[:, self.n - 1: ]
		out = self.embedding(x)

		out = out.permute(0,2,1)

		if self.n_gram_model == 'cnn':
			# print("Before cnn:", out.size()) # torch.Size([30, 768, 119])
			out = self.conv(out)  #batch x max_length (n - 1) x hidden
			# print("After cnn:", out.size()) # torch.Size([30, 100, 115])

		elif self.n_gram_model == 'bert':
			# call a function that applies bert and returns output same form as the cnn
			out = self.apply_transformer(out, lengths)


		out= self.relu(out)
		out = self.drop(out)

		for i in range(len(self.linears)-1):
			out = self.linears[i](out)
			out= self.relu(out)
			out = self.drop(out)

		# we do not apply dropout on the last layer
		out = self.linears[-1](out)

		out= self.relu(out)


		# batch x max_length  - (n-1)x out_size


		mask = mask.unsqueeze(1).repeat(1, self.sparse_dimensions,1).float()
		out = (mask * out).sum(2) / lengths.unsqueeze(1)
		# batch x max_length - (n-1) x out_size
		return out

	def init_transformer(self):

		config = transformers.BertConfig(vocab_size = self.vocab_size, hidden_size = self.embedding_dim, num_hidden_layers = 1,
										num_attention_heads = 4, intermediate_size = self.embedding_dim * 4 , max_position_embeddings = self.n)

		self.transformer = transformers.BertModel(config)
		self.resize_linear = torch.nn.Linear(self.embedding_dim, self.hidden_sizes[0])


	def apply_transformer(self, input, lengths):

		max_length = input.size(-1)

		n_gram_outputs = []

		# mimic sliding window behavior
		for i in range(max_length - self.n + 1):
			# n-gram input for each sample in the batch, shifted by i
			temp_in = input[:,:,  i: i + self.n].permute(0, 2, 1)
			# forward it to the transformer model
			last_hidden_state, pooler_output = self.transformer(inputs_embeds = temp_in)
			# calculate average over outputs of each n-gram
			last_hidden_state = last_hidden_state.mean(dim = 1)
			# apply resize linear, so that the hidden representation is ready to be forwarded to the first linear of the self.hidden_sizes
			last_hidden_state = self.resize_linear(last_hidden_state)
			# append slidding window averaged results to a list of outputs
			n_gram_outputs.append(last_hidden_state)

		return torch.stack(n_gram_outputs, dim=-1) 
