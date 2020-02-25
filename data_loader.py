from dataset import MSMarcoTrain, MSMarcoDev
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd

def get_data_loaders(dataset_path, train_batch_size, val_batch_size, tokenizer):

    dataloaders = {}
    # dataloaders['train'] = DataLoader(MSMarco(dataset_path, 'train'),
    #     batch_size=train_batch_size, collate_fn=collate_fn_padd)
    dataloaders['test'] =  DataLoader(MSMarcoDev(dataset_path),
        batch_size=val_batch_size, collate_fn=collate_fn_padd)

    return dataloaders
