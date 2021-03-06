import torch
import transformers
from transformers.activations import gelu, gelu_new

#from enlp.models.bert_no_layer_norm import BertModelNoOutLayerNorm
from transformers import BertTokenizerFast
import math

class BERT_inter(torch.nn.Module):
	def __init__(self, hidden_size = 256, num_of_layers = 2, num_attention_heads = 4, input_length_limit = 512,
			vocab_size = 30522, embedding_parameters = None, params_to_copy = {}):
		super(BERT_inter, self).__init__()


		self.model_type = "bert-interaction"
		output_size = 2

		if embedding_parameters is not None:
			# adjust hidden size and vocab size
			hidden_size = embedding_parameters.size(1)
			vocab_size = embedding_parameters.size(0)



# not sure if this is correct or if it shoulds stay, probably its ok ---->>>>
		# traditionally the intermediate_size is set to be 4 times the size of the hidden size
		intermediate_size = hidden_size*4

		# set up the Bert config
		config = transformers.BertConfig(vocab_size = vocab_size, hidden_size = hidden_size, num_hidden_layers = num_of_layers,
										num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = input_length_limit)

		self.encoder = transformers.BertModel(config)

		self.last_linear = torch.nn.Linear(hidden_size, hidden_size)

		self.output_linear = torch.nn.Linear(hidden_size, output_size)
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

	def forward(self, input_ids, attention_masks, token_type_ids, output_attentions=False, output_hidden_states=False):
		all_out = self.encoder(input_ids = input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
		print(output_attentions, output_hidden_states)
		last_hidden_state = all_out.last_hidden_state

		# extract hidden representation of the 1st token, that is the CLS special token
		cls_hidden_repr = last_hidden_state[:,0]

		out = self.last_linear(cls_hidden_repr)

		out = torch.tanh(out)

		out = self.output_linear(out)
		if output_hidden_states or output_attentions:
			return [out, all_out]
		return out
