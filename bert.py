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
