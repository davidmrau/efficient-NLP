import numpy as np
import torch

import transformers




class BERT_based(torch.nn.Module):
    def __init__(self, hidden_size = 256, num_of_layers = 2, sparse_dimensions = 10000, vocab_size = 30522, num_attention_heads = 4, input_length_limit = 150, embedding_path = 'bert', word2idx = None, pooling_method = "CLS", device='cpu'):
        super(BERT_based, self).__init__()

        # if we use pretrained BERT embeddings, we have to use the same hidden size
        if embedding_path != '':

            if embedding_path == "bert":
                embeddings = self.get_pretrained_BERT_embeddings()
            else:
                embeddings = load_glove_embeddings(embedding_path, word2idx, device)

            hidden_size = embeddings.size(1)

        # traditionally the intermediate_size is set to be 4 times the size of the hidden size
        intermediate_size = hidden_size*4
        # we save the pooling methid, will need it in forward
        self.pooling_method = pooling_method
        # set up the Bert-like config
        config = transformers.BertConfig(vocab_size = vocab_size, hidden_size = hidden_size, num_hidden_layers = num_of_layers,
                                        num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = input_length_limit)
        # Initialize the Bert-like encoder
        self.encoder = transformers.BertModel(config)
        self.relu = torch.nn.ReLU()
        self.device = device

        if pretrained_embeddings:
            self.encoder.embeddings.word_embeddings.weight = embeddings

        # the last linear of the model that projects the dense space to sparse space
        self.sparse_linear = torch.nn.Linear(hidden_size, sparse_dimensions)

    def get_pretrained_BERT_embeddings(self):
        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        return bert.embeddings.word_embeddings.weight

    def forward(self, input, lengths):
        attention_masks = torch.zeros(input.size(0), lengths.max().int().item()).to(self.device)

        for i in range(lengths.size(0)):
            attention_masks[i, : lengths[i].int()] = 1

        last_hidden_state, pooler_output = self.encoder(input_ids = input, attention_mask=attention_masks)

        # aggregate output of model, to a single representation
        if self.pooling_method == "CLS":
            encoder_output = pooler_output

        elif self.pooling_method == "AVG":
            # not taking into account outputs of padded input tokens
            encoder_output = (last_hidden_state * attention_masks.unsqueeze(-1).repeat(1,1,self.encoder.config.hidden_size).float()).sum(dim = 1)
            # dividing each sample with its actual lenght for proper averaging
            encoder_output = encoder_output / attention_masks.sum(dim = -1).unsqueeze(1)

        elif self.pooling_method == "MAX":
            # not taking into account outputs of padded input tokens
            # taking the maximum activation for each hidden dimension over all sequence steps
            encoder_output = (last_hidden_state * attention_masks.unsqueeze(-1).repeat(1,1,self.encoder.config.hidden_size).float()).max(dim = 1)[0]

        output = self.sparse_linear(encoder_output)
        output = self.relu(output)
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
# (word_embeddings): Embedding(30522, 768, padding_idx=0)
# (position_embeddings): Embedding(512, 768)
# (token_type_embeddings): Embedding(2, 768)
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
