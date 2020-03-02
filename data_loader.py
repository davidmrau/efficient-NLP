from dataset import MSMarco, MSMarcoInference
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd, read_pickle


def get_data_loaders(dataset_path, batch_size, debug=False):

    dataloaders = {}
    docs_offset_list = read_pickle(f'{dataset_path}/collection.tokenized.tsv.offset_list.p')
    dataloaders['train'] = DataLoader(MSMarco(dataset_path, 'train', docs_offset_list, debug=debug),
    batch_size=batch_size, collate_fn=collate_fn_padd)
    dataloaders['val'] =  DataLoader(MSMarco(dataset_path, 'dev', docs_offset_list, debug=debug),
        batch_size=batch_size, collate_fn=collate_fn_padd)

    return dataloaders


def get_data_loaders_online(dataset_path, batch_size, debug=False):

    dataloaders = {}

    dataloaders['val'] = DataLoader(MSMarcoInference(f'{dataset_path}/queries.dev.tsv.p',f'{dataset_path}/queries.dev.tsv.ids.p',  debug=debug),
    batch_size=batch_size, collate_fn=collate_fn_padd)
    dataloaders['test'] =  DataLoader(MSMarcoInference(f'{dataset_path}/queries.eval.tsv.p', f'{dataset_path}/queries.eval.tsv.ids.p', debug=debug),
        batch_size=batch_size, collate_fn=collate_fn_padd)
    return dataloaders


def get_data_loaders_offline(dataset_path, batch_size, debug=False):
    dataloaders = {}
    dataloaders['docs'] = DataLoader(MSMarcoInference(f'{dataset_path}/collection.tsv.p',f'{dataset_path}/collection.tsv.ids.p' ,debug=debug),
    batch_size=batch_size, collate_fn=collate_fn_padd)

    return dataloaders
