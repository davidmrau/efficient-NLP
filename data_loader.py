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



def get_data_loaders_online(dataset_path, batch_size, debug=False):

	dataloaders = {}


	queries_test = FileInterface(f'{dataset_path}/queries.eval.tsv.p')
	queries_val = FileInterface(f'{dataset_path}/queries.dev.tsv.p')
	dataloaders['val'] = DataLoader(MSMarcoInference(queries_val),
	batch_size=batch_size, collate_fn=collate_fn_padd)
	dataloaders['test'] =  DataLoader(MSMarcoInference(queries_test),
		batch_size=batch_size, collate_fn=collate_fn_padd)
	return dataloaders


def get_data_loaders_offline(dataset_path, batch_size, debug=False):
	dataloaders = {}
	debug_str = '' if not debug else '.debug'
	docs = FileInterface(f'{dataset_path}/qidpidtriples.{split}.full{debug_str}.tsv', map_index=False)
	dataloaders['docs'] = DataLoader(MSMarcoInference(docs, debug),
	batch_size=batch_size, collate_fn=collate_fn_padd)

	return dataloaders
