from dataset import MSMarco, MSMarcoInference
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd, read_pickle
from file_interface import FileInterface


def get_data_loaders(dataset_path, batch_size, debug=False):

	dataloaders = {}
	# docs_offset_list = read_pickle(f'{dataset_path}/collection.tokenized.tsv.offset_list.p')
	dataloaders['train'] = DataLoader(MSMarco(dataset_path, 'train', debug=debug),
	batch_size=batch_size, collate_fn=collate_fn_padd, shuffle=True)
	dataloaders['val'] =  DataLoader(MSMarco(dataset_path, 'dev', debug=debug),
		batch_size=batch_size, collate_fn=collate_fn_padd)

	return dataloaders
