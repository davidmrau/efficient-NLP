from dataset import MSMarco
from torch.utils.data import DataLoader
from os import path
from utils import collate_fn_padd

def get_data_loaders(dataset_path, train_batch_size, val_batch_size, tokenizer):

    dataloaders = {}
    dataloaders['train'] = DataLoader(MSMarco(dataset_path, 'train'),
        batch_size=train_batch_size, collate_fn=collate_fn_padd)
    dataloaders['test'] =  DataLoader(MSMarco(dataset_path, 'eval'),
        batch_size=val_batch_size, collate_fn=collate_fn_padd)
    # dataloaders['val'] =   DataLoader(ELI5(dataset_path+'val.p'), batch_size=val_batch_size, collate_fn=collate_fn_padd)

    return dataloaders
