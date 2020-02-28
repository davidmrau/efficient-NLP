import numpy as np
import torch
import torch.nn as nn


import transformers


# MODEL_CLASSES = {
#     "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
#     "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
#     "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
#     "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
#     "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
#     "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
# }
class BERT_Based(nn.Module):
    def __init__(self):
        super(BERT_Based, self).__init__()

        self.BERT_encoder = transformers.BertModel.from_pretrained('bert-base-uncased') # bert-base-uncased,  bert-large-cased

            # self.encoder._resize_token_embeddings(new_num_tokens = vocab_size)

        self.sparse_linear = nn.Linear(768, 10000)
        self.out_linear = nn.Linear(10000, 1)

    def forward(self, input, lengths):

        attention_masks = torch.zeros(input.size(0), max(lengths))

        for i in range(input.size(0)):
            attention_masks[i, : lengths[i]] = 1


        # ["CLS"] special token needs to be put at the begining of each sequence (on daa preprocessing)
        _, out = self.BERT_encoder(input_ids = input, attention_mask=attention_masks)

        out = self.sparse_linear(out)

        out = self.out_linear(out)

        return out


import torch
import transformers




class BERT_based(torch.nn.Module):
    def __init__(self, hidden_sizes = [256,256,10000], vocab_size = 30522, num_attention_heads = 4, input_length_limit = 150, pretrained_embeddings = False, pooling_method = "CLS"):
        super(BERT_based, self).__init__()

        hidden_size = hidden_sizes[0]
        sparse_linear_hidden_size = hidden_sizes[-1]
        num_hidden_layers = len(hidden_sizes) - 1

        intermediate_size = hidden_size*4

        self.pooling_method = pooling_method

        config = transformers.BertConfig(vocab_size_or_config_json_file = vocab_size, hidden_size = hidden_size, num_hidden_layers = num_hidden_layers,
                                        num_attention_heads = num_attention_heads, intermediate_size = intermediate_size, max_position_embeddings = input_length_limit)

        self.encoder = transformers.BertModel(config)

        if pretrained_embeddings:
            self.encoder.embeddings.word_embeddings.weight = self.get_pretrained_BERT_embeddings()

        self.sparse_linear = torch.nn.Linear(hidden_size, sparse_linear_hidden_size)

    def get_pretrained_BERT_embeddings(self):
        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        return bert.embeddings.word_embeddings.weight


    def forward(self, input, lengths):


        attention_masks = torch.zeros(input.size(0), lengths.max().int().item())

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
            encoder_output = (last_hidden_state * attention_masks.unsqueeze(-1).repeat(1,1,self.encoder.config.hidden_size).float()).max(dim = 1)

        output = self.sparse_linear(encoder_output)

        return output



# input =
hidden_sizes =[256,256,10000]
model = BERT_based(hidden_sizes, pooling_method = "AVG")

output = model(torch.tensor( [[5,6,7,0], [8,9,105,1555]]), lengths = torch.FloatTensor([3,4]) )

print(output.size())
