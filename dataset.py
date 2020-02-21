
import numpy as np
from os import path
import torch
from torch.utils import data
from utils import *
import json
import re
from torchtext.data.utils import get_tokenizer


class ELI5(data.Dataset):

    def __init__(self, query_path, documents_path, scores_dict_path, tokenizer):
        self.query_ids, self.queries = read_data(query_path)
        self.doc_ids, self.docs = read_data(documents_path)
        self.docs_dict = {id: d for id, d in zip(self.doc_ids, self.docs)}
        self.scores_dict = read_json(scores_dict_path)
        self.tokenizer = tokenizer
        self.tokenizer_spacy = get_tokenizer("spacy")
    def __len__(self):
        length = self.queries.shape[0]
        return length

    def text2id(self, text):
        return [self.tokenizer.convert_tokens_to_ids(word) for word in self.tokenizer_spacy(text.lower())]

    def __getitem__(self, index):
        query_id = self.query_ids[index]
         # sample relevant doc for query
        doc_ids, doc_scores = self.scores_dict[query_id]
        # normalize scores
        doc_scores = np.array(doc_scores)
        doc_scores = doc_scores / doc_scores.sum()

        # trick to sample index rather than docid to obain associated probabilites easy
        index_docs = np.arange(len(doc_ids))

        #sample doc 1
        doc1_idx = np.random.choice(index_docs, p=doc_scores)
        # get doc id
        doc1_id = doc_ids[doc1_idx]
        # get score for doc1
        doc1_score = doc_scores[doc1_idx]

        # flip coin whether to get two relevant docs or relevant + non relevant
        if np.random.random() >= 0.5:
            #sample doc 1
            doc2_idx = np.random.choice(index_docs, p=doc_scores)
            # get doc id
            doc2_id = doc_ids[doc2_idx]
            # get score for doc1
            doc2_score = doc_scores[doc2_idx]
            # determine target
            target = 1 if (doc1_score >= doc2_score) else -1
        else:
            # non_relevant
            doc2_id = np.random.choice(self.doc_ids)
            target = -1

        query = self.queries[index]
        doc1 = self.docs_dict[doc1_id]
        doc2 = self.docs_dict[doc2_id]
        query = self.text2id(query)
        doc1 = self.text2id(doc1)
        doc2 = self.text2id(doc2)
        return [query, doc1, doc2], target
