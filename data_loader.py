from dataset import MSMarcoTrain, MSMarcoDev
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd

def get_data_loaders(dataset_path, batch_size, tokenizer, debug=False):

    dataloaders = {}
    dataloaders['train'] = DataLoader(MSMarcoTrain(dataset_path, debug=debug),
    batch_size=batch_size, collate_fn=collate_fn_padd)
    dataloaders['test'] =  DataLoader(MSMarcoDev(dataset_path, debug=debug),
        batch_size=batch_size, collate_fn=collate_fn_padd)

    return dataloaders
