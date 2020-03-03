
from data_loader import get_data_loaders_offline
import torch
import hydra
from hydra import utils
from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence

# from transformers import BertConfig, BertForPreTraining, BertTokenizer


@hydra.main(config_path='config.yaml')

def exp(cfg):
	# Initialize an Inverted Index object
	ii = InvertedIndex(vocab_size = cfg.embedding_dim, num_of_workers=cfg.num_of_workers_index)
	# initialize the index
	ii.initialize_index()

	orig_cwd = utils.get_original_cwd() + '/'

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')


	# open file
	debug_str = '' if not debug else '.debug'
	filename = f'{dataset_path}/qidpidtriples.{split}.full{debug_str}.tsv'
	file = open(filename, 'r')


	line = file.readline()

	# until we have read the complete file
	while(True):

		# read a number of lines equal to batch_size
		batch_ids = []
		batch_data = []
		while(line and ( len(batch_ids) < cfg.batch_size) ):

			# if line was last line then break!

			# getting position of first ' ' that separates the doc_id and the begining of the token ids
			delim_pos = line.find(' ')
			# extracting the id
			id = line[:delim_pos]
			# extracting the token_ids and creating a numpy array
			tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')
			batch_ids.append(id)
			batch_data.append(tokens_list)

			line = file.readline()


		batch_lengths = torch.FloatTensor([len(d) for d in batch_data])
		#padd data along axis 1
		batch_data = pad_sequence(batch_data,1).long()

		logits = model(batch_data.to(device), batch_lengths.to(device))

		ii.add_docs_to_index(ids.cpu().numpy(), logits.cpu())


		if not line:
			break


	# sort the posting lists
	ii.sort_posting_lists()


	# dataloaders['docs'] = DataLoader(MSMarcoInference(f'{dataset_path}/qidpidtriples.{split}.full{debug_str}.tsv'),
	# batch_size=batch_size, collate_fn=collate_fn_padd)

	#
	# dataloaders = get_data_loaders_offline(orig_cwd + cfg.dataset_path, cfg.batch_size, debug=cfg.debug)
	# # load model
	# model = torch.load(orig_cwd + cfg.model_path)
	#
	# # for each document in the collection, pass it through the model, and use its sparse output vector for indexing
	# for data, ids, lengths in dataloaders['docs']:
	# 	logits = model(data.to(device), lengths.to(device))
	#

		ii.add_docs_to_index(ids.cpu().numpy(), logits.cpu())

	# sort the posting lists
	ii.sort_posting_lists()

if __name__ == "__main__":
	exp()
