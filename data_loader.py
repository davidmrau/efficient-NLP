from dataset import MSMarcoTrain, MSMarcoDev
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd

def get_data_loaders(dataset_path, train_batch_size, val_batch_size, tokenizer, debug_mode=False):

    dataloaders = {}
    dataloaders['train'] = DataLoader(MSMarcoTrain(dataset_path, debug_mode=debug_mode),
    batch_size=train_batch_size, collate_fn=collate_fn_padd)
    dataloaders['test'] =  DataLoader(MSMarcoDev(dataset_path, debug_mode=debug_mode),
        batch_size=val_batch_size, collate_fn=collate_fn_padd)

    return dataloaders
