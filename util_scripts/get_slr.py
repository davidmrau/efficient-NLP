import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from enlp.dataset import Sequential, collate_fn_padd_single
from enlp.utils import load_model

""" Run online inference (for test set) without inverted index
"""



def slr(cfg):

	if not cfg.disable_cuda and torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')


	model = load_model(cfg, cfg.model_folder, device)

	dataset = Sequential(cfg.input, tokenize=cfg.tokenize)
	dataloader =  DataLoader(dataset, batch_size=64, collate_fn=collate_fn_padd_single)

	with torch.no_grad():
		model.eval()
		reprs = list()
		ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in dataloader:
			repr_ = model(batch_data_d.to(device), batch_lengths_d.to(device))
			reprs.append(repr_.detach().cpu().numpy())
			ids += batch_ids_d
		reprs = np.vstack(reprs)

		out_name = f'{cfg.input}_reprs.tsv'

		with open(out_name, 'w') as out:

			for id_, repr_ in zip(ids, reprs):
				repr_str = ' '.join([str(r) for r in repr_])
				out.write(f'{id_}\t{repr_str}\n')

if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# getting model config
	if not cl_cfg.model_folder or not cl_cfg.input or cl_cfg.tokenize == None:
		raise ValueError("usage: get_slr.py model_folder=MODEL_FOLDER input=INPUT_PATH tokenize=True|False")
	# getting model config
	cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
	# merging both
	cfg = OmegaConf.merge(cfg_load, cl_cfg)
	slr(cfg)
