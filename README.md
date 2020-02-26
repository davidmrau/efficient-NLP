# efficient-NLP

## Download pre-processed MSMARCO

Download the pre-processed dataset: https://drive.google.com/file/d/19PDLjGehBKwTa2Oa5a1wqK8X5vhrPKp4/view?usp=sharing
We have tokenized all queries [1] and documents [2] and translated them into token-ids following the bert vocabulary ('bert-base-uncased') using the transformer library [3].


The data conists of:


**Queries** are split into:
- queries.train.tsv.p
- queries.dev.tsv.p
- queries.eval.tsv.p

Every file contains a dict:
```
{qid_1: [token_1, ..., token_n], ..., qid_d: [token_1, ..., token_m]}
```

**Docs** can be found in:
- collection.tsv.p

Contains a dict: 
```
{did_1: [token_1, ..., token_n], ..., did_d: [token_1, ..., token_m]}
```

**Training triples**:

- qidpidtriples.train.full.tsv

Contains triplets: 
```
q_id  relevant_doc_id non_relevant_doc_id
```

**Ground truth**:

- qrels.train.tsv
- qrels.dev.tsv

Contains:
```
q_id  0 relevant_doc_id 1
```


extract data using ```tar -xzf data.tar.gz```


[1] https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz
[2] https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
[3] https://huggingface.co/transformers/main_classes/tokenizer.html
