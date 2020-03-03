
from dataset import MSMarcoSequential
import torch
import hydra
from hydra import utils
from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from snrm import SNRM
from utils import str2lst
import transformers
from transformers import BertTokenizer
# from transformers import BertConfig, BertForPreTraining, BertTokenizer


@hydra.main(config_path='config.yaml')

def exp(cfg):
	# Initialize an Inverted Index object
	ii = InvertedIndex(vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)
	# initialize the index
	ii.initialize_index()

	orig_cwd = utils.get_original_cwd() + '/'

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	# define which embeddings to load, depending on params
	if cfg.embedding == 'glove':
		embedding_path = orig_cwd + cfg.glove_embedding_path
	elif cfg.embedding == 'bert':
		embedding_path = 'bert'

	# open file
	debug_str = '' if not cfg.debug else '.debug'
	filename = f'{orig_cwd}{cfg.dataset_path}/collection.tokenized.tsv'

	# load BERT's BertTokenizer
	tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
	# use BERT's word to ID
	word2idx = tokenizer.vocab
	print('Initializing model...')

	ms = MSMarcoSequential(filename, cfg.batch_size)
	model = SNRM(hidden_sizes=str2lst(str(cfg.snrm.hidden_sizes)),
	sparse_dimensions = cfg.sparse_dimensions, n=cfg.snrm.n, embedding_path=embedding_path,
	word2idx=word2idx, dropout_p=cfg.snrm.dropout_p, debug=cfg.debug, device=device).to(device)
	i = 0
	for batch_ids, batch_data, batch_lengths in ms.batch_generator():
		# print(batch_data)
		logits = model(batch_data.to(device), batch_lengths.to(device))
		i += 1
		ii.add_docs_to_index(batch_ids, logits.cpu())
		print(i)


	# sort the posting lists
	ii.sort_posting_lists()



if __name__ == "__main__":
	exp()
