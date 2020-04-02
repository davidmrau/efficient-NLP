
## Running the complete pipeline

```
# 1. clone git repository
git clone https://github.com/davidmrau/efficient-NLP

# 2. move to slurm_scripts dir
cd efficient-NLP/slurm_scripts

# 3. download preprocessed tokenized data
bash startup.sh

# 4. [OPTIONAL] add your slurm header to main.sh

# 5. run hyper-param search
bash hyper-tf.sh
```

### Training
**Training Samples**
We train on the tripplets provided by the MS Marco dataset (q_id, relevant_doc, irrelevant_doc), while randomizing order of docs. We parse the training samples in suffled order, and we evaluate the model every 100.000 training steps.

**Training loss**
The Hinge Loss is used for training, so that the model can give higher score to the relevant document.
L1 loss is also taken into account, for making the output vector as sparse as possible (Following Zamani et. all 2018 [4])
L1 Loss is divided by the number of dimensions of the output space, andmultiplied by l1_scalar parameter that can be set as a paramter on training.
The ultimate loss is **L = Hinge_loss + l1_scalar*L1_loss**

**Evaluation Samples**
For evaluating the model during training, we use the queries from the Dev set. For each query, a relevant passage is provided. We sample a random passage which we assume to be irrelevant and we evaluate how well the model is able to identify the relevant passage.

**Training Termination**
The training is terminated if 100 training epochs are completed, or if the model has not improved its performance on the evaluation set after 5 consequent epochs (patience=5).

### Building the Inverted Index
After the training is terminated, we create an inverted index (for the [Test Set](#test-set)). The inverted index consists of posting lists for each output dimension (latent term). We do this by passing each document through the model, and given the model's activations, we assign it to the appropriate posting lists.

**Model's Behaviour Overview**
Creating the Inverted Index, results into saving two plots into files.
num_latent_terms_per_doc.pdf  : gives an overview of the sparsity of the model
num_docs_per_latent_term.pdf  : given an overview of the distribution of documetns over the latent terms / posting lists.

### Evaluation
We propagate the queries of the [Test Set ](#test-set) from the file {queries_filename} through our model in order to get asparse representation, and retrieve relevant sentences using the inverted index. We save the top-10 results for each query to a file (ranking_results.{queries_filename}). Then we use the official MS Marco evaluation script (ms_marco_eval.py) in order to get the MRR metric results, that we also save to a file (metrics.{queries_filename}).


### File Structure (After Succesfully Completed Pipeline)
"experiments" directory is being created under "efficient-NLP" directory. The experiments directory has one directory for each experiment, the name of which depends on the hyper-parameters of the experiments. Inside the directory of one experiment, we save:
* best_model.model (trained model)
* config.yaml     (parameters of the execution)
* Inverted_Index (directory that containes one file for each posting list / latent term)
* latent_terms_per_doc_dict.p (pickle python dictionary, containing the number of latent terms for each document key = str(doc_id)
* posting_lists_lengths (pickle python dictionary, containing the number of documents for each posting list, equal to the number of lines per posting list file)
* num_latent_terms_per_doc.pdf (histogram displaying the number of latent terms for the documents fo the dataset)
* num_docs_per_latent_term.pdf (plot showing the number of documents for each posting list (Sorted)
* ranking_results.{queries_filename} (top 10 results per query, accompanies by their rank)
* metrics.{queries_filename} (final metric results compared to ground truth file, obtained from official MS Marco script)



### Dataset: Currently Iplemented for MS Marco Passage Full-Ranking task:

**Preprocessed Data**
We have tokenized all queries [1] and documents [2] and translated them into token-ids following the bert vocabulary ('bert-base-uncased') using the transformer library [3]. Further, we truncated all documents to a maximum number of tokens of 150.

The data.tar.gz consists:


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

**Test Set** <a name="test-set"></a>
Test set obtained from https://microsoft.github.io/TREC-2019-Deep-Learning/ (Passage Ranking Dataset)
This test set containes 200 queries (msmarco-test2019-queries.tsv), their relevant document (passage) ids, and 1000 documents for each (msmarco-passagetest2019-top1000.tsv), creating a document set of 200*1000 (by running get_docs_from_test.py). We build the inverted index and run the evaluation only for these, in order to get a representative performance.

**Embeddings**:
Setting the argument "embedding" to "bert" or "glove" while running main, loads the according embeddings weights to the model (by typing ```embedding=bert```).
The glove embedding file that is used to initialize the embedding layer. It will be extracted to data/embedding/glove.6B.300d.txt.





[1] https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz

[2] https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz

[3] https://huggingface.co/transformers/main_classes/tokenizer.html

[4] https://dl.acm.org/doi/10.1145/3269206.3271800


