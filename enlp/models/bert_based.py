import torch
import transformers
#from transformers.activations import gelu, gelu_new

#from enlp.models.bert_no_layer_norm import BertModelNoOutLayerNorm
from transformers import BertTokenizerFast
import math

def Delu(x):
	return x * 0.5 * (1 + torch.erf(x / math.sqrt(0.3)))

class BERT_based(torch.nn.Module):
	def __init__(self, hidden_size = 256, num_of_layers = 2, sparse_dimensions = 1000, num_attention_heads = 4, input_length_limit = 150,
			vocab_size = 30522, embedding_parameters = None, pooling_method = "CLS", large_out_biases = False, layer_norm = True,
			act_func="relu", params_to_copy = {}):
		super(BERT_based, self).__init__()


		self.model_type = "representation-based"

		if embedding_parameters is not None:
			# adjust hidden size and vocab size
			hidden_size = embedding_parameters.size(1)
			vocab_size = embedding_parameters.size(0)

		# in case we are agregating the hidden representation by using the hidden representation of the CLS token,
		# we need to add this token at the beginning of each input, and adjust the input lenght limit
		if pooling_method == "CLS":
			input_length_limit += 1
			# in case we are using the pretrained BERT embeddings, its best if we use the coresponding CLS token ID
			if embedding_parameters is not None and hidden_size == 768 and vocab_size == 30522:
				self.cls_token_id = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased').encode("[CLS]")[1]
				print("Using pretrained BERT embeddigns, retrieving [CLS] token id.")
				# otherwise, we add the CLS token to the vocabulary
			else:
				print("Not using pretrained BERT embeddings, [CLS] token is the last word of vocab")
				self.cls_token_id = vocab_size - 1



		# traditionally the intermediate_size is set to be 4 times the size of the hidden size
		intermediate_size = hidden_size*4
		# we save the pooling methid, will need it in forward
		self.pooling_method = pooling_method
		# set up the Bert-like config
		config = transformers.BertConfig(vocab_size = vocab_size, hidden_size = hidden_size, num_hidden_layers = num_of_layers,
										num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = input_length_limit)
		# Initialize the Bert-like encoder
		if layer_norm:
			self.encoder = transformers.BertModel(config)
		else:
			raise NotImplementedError()
			#self.encoder = BertModelNoOutLayerNorm(config)

		if act_func == "relu":
			self.act_func = torch.nn.ReLU()
		elif act_func == "gelu":
			self.act_func = gelu
		elif act_func == "gelu_new":
			self.act_func = gelu_new
		elif act_func == "delu":
			self.act_func = Delu
		else:
			raise ValueError("Activation Function, was not set properly!")

		if embedding_parameters is not None:
			# copy loaded pretrained embeddings to model
			self.encoder.embeddings.word_embeddings.weight = torch.nn.Parameter(embedding_parameters)

		# copy all specified parameters
		for param in params_to_copy:

			param_splitted = param.split(".")

			item = self.encoder.__getattr__(param_splitted[0])

			for p in param_splitted[1: -1]:
				item  = item.__getattr__(p)

			last_item = param_splitted[-1]

			setattr(item, last_item, params_to_copy[param])

		# the last linear of the model that projects the dense space to sparse space
		self.sparse_linear = torch.nn.Linear(hidden_size, sparse_dimensions)

		if large_out_biases:
			self.sparse_linear.bias = torch.nn.Parameter(torch.ones(sparse_dimensions) * 3 )

	def get_cls_token_id(self, ):
		bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


	def forward(self, input, lengths):

		# in case we are aggregating using the CLS token, we add the "[CLS]" token id at the beginning of each input, and adjust the lengths
		if self.pooling_method == "CLS":
			# add CLS_token_id column at the beginning of the input
			cls_column = torch.ones_like(input[:,0]).unsqueeze(-1) * self.cls_token_id
			input = torch.cat([cls_column, input], dim = 1)
			# increase lengths by 1, accordingly
			lengths += 1

		attention_masks = torch.zeros_like(input)

		for i in range(lengths.size(0)):
			attention_masks[i, : lengths[i].int()] = 1

		if next(self.parameters()).is_cuda:
			attention_masks = attention_masks.cuda()


		last_hidden_state, pooler_output = self.encoder(input_ids = input, attention_mask=attention_masks)

		# aggregate output of model, to a single representation
		if self.pooling_method == "CLS":
			# encoder_output = pooler_output
			encoder_output = last_hidden_state[:,0,:]

		elif self.pooling_method == "AVG":
			attention_masks = attention_masks.float()
			# not taking into account outputs of padded input tokens
			encoder_output = (last_hidden_state * attention_masks.unsqueeze(-1).repeat(1,1,self.encoder.config.hidden_size)).sum(dim = 1)
			# dividing each sample with its actual lenght for proper averaging
			encoder_output = encoder_output / attention_masks.sum(dim = -1).unsqueeze(1)

		elif self.pooling_method == "MAX":
			attention_masks = attention_masks.float()
			# not taking into account outputs of padded input tokens
			# taking the maximum activation for each hidden dimension over all sequence steps
			encoder_output = (last_hidden_state * attention_masks.unsqueeze(-1).repeat(1,1,self.encoder.config.hidden_size)).max(dim = 1)[0]

		output = self.sparse_linear(encoder_output)

		# Always using the ReLU activation function while evaluating
		if self.training:
			output = self.act_func(output)
		else:
			output = torch.nn.functional.relu(output)

		return output


	#
	# def get_optimizer(self, n_train_batches = 1000000):
	#
	#     if self.args.max_steps > 0:
	#         t_total = self.args.max_steps
	#         self.args.num_train_epochs = self.args.max_steps // (n_train_batches // self.args.gradient_accumulation_steps) + 1
	#     else:
	#         t_total = n_train_batches // self.args.gradient_accumulation_steps * self.args.num_train_epochs
	#
	#
	#         # optimezer, scheduler# Prepare optimizer and schedule (linear warmup and decay)
	#     no_decay = ['bias', 'LayerNorm.weight']
	#     optimizer_grouped_parameters = [
	#         {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
	#         {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
	#         ]
	#     self.optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
	#     self.scheduler = transformers.WarmupLinearSchedule(self.optimizer, warmup_steps=self.args.warmup_steps, t_total=t_total)
	#
	#

#
#
# # input =
# hidden_sizes =[256,256,10000]
# model = BERT_based(hidden_sizes, pooling_method = "MAX")
#
# model.set_encoder_to_N_first_layers_of_pretrained_BERT( N = 5)
#
# exit()
# output = model(torch.tensor( [[5,6,7,0], [8,9,105,1555]]), lengths = torch.FloatTensor([3,4]) )
#
# print(output.size())
#
# -----------------------

# BertModel:
#
# (embeddings): BertEmbeddings(
# (word_embeddings): Embeddings(30522, 768, padding_idx=0)
# (position_embeddings): Embeddings(512, 768)
# (token_type_embeddings): Embeddings(2, 768)
# (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
# (dropout): Dropout(p=0.1, inplace=False)
# )
# (encoder): BertEncoder(
# (layer): ModuleList(
#   (0): BertLayer(
#     (attention): BertAttention(
#       (self): BertSelfAttention(
#         (query): Linear(in_features=768, out_features=768, bias=True)
#         (key): Linear(in_features=768, out_features=768, bias=True)
#         (value): Linear(in_features=768, out_features=768, bias=True)
#         (dropout): Dropout(p=0.1, inplace=False)
#       )
#       (output): BertSelfOutput(
#         (dense): Linear(in_features=768, out_features=768, bias=True)
#         (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#         (dropout): Dropout(p=0.1, inplace=False)
#       )
#     )
#     (intermediate): BertIntermediate(
#       (dense): Linear(in_features=768, out_features=3072, bias=True)
#     )
#     (output): BertOutput(
#       (dense): Linear(in_features=3072, out_features=768, bias=True)
#       (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#       (dropout): Dropout(p=0.1, inplace=False)
#     )
#   )

# class BertPooler(nn.Module):
#     def __init__(self, config):
#         super(BertPooler, self).__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.activation = nn.Tanh()
#
#     def forward(self, hidden_states):
#         # We "pool" the model by simply taking the hidden state corresponding
#         # to the first token.
#         first_token_tensor = hidden_states[:, 0]
#         pooled_output = self.dense(first_token_tensor)
#         pooled_output = self.activation(pooled_output)
#         return pooled_output


# (pooler): BertPooler(
#     (dense): Linear(in_features=768, out_features=768, bias=True)
#     (activation): Tanh()
#   )
