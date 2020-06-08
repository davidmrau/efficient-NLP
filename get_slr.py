import torch
from omegaconf import OmegaConf
from utils import load_model, collate_fn_padd_single
from torch.utils.data import DataLoader, IterableDataset
from tokenizer import Tokenizer
import numpy as np
""" Run online inference (for test set) without inverted index
"""


class Sequential(IterableDataset):
	def __init__(self, fname, tokenize=False):

		# open file
		self.file_ = open(fname, 'r')
		self.tokenize = tokenize

		self.tokenizer = Tokenizer(tokenizer = 'glove', max_len = 150, stopwords='lucene', remove_unk = True, unk_words_filename=None)

	def __iter__(self):
			for line in self.file_:
				# getting position of '\t' that separates the doc_id and the begining of the token ids
				delim_pos = line.find('\t')
				# extracting the id
				id_ = line[:delim_pos]
				print('id', id_)
				# extracting the token_ids and creating a numpy array
				if self.tokenize:
					tokens_list = self.tokenizer.encode(line[delim_pos+1:])
				else:
					tokens_list = np.fromstring(line[delim_pos+1:], dtype=int, sep=' ')

				yield [id_, tokens_list]

def slr(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')


	model = load_model(cfg, cfg.model_folder, device)

	dataset = Sequential(cfg.input, tokenize=cfg.tokenize)
	dataloader =  DataLoader(dataset, batch_size=8, collate_fn=collate_fn_padd_single)

	with torch.no_grad():
		model.eval()
		reprs = list()
		ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in dataloader:
			repr_ = model(batch_data_d.to(device), batch_lengths_d.to(device))
			reprs.append(repr_.detach().cpu().numpy())
			ids += batch_ids_d
		print(reprs)


if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.input or not cl_cfg.tokenize:
		raise ValueError("usage: get_slr.py model_folder=MODEL_FOLDER input=INPUT_PATH tokenize=True|False")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	slr(cfg)
