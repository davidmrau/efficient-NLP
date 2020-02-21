from dataset import ELI5
from torch.utils.data import DataLoader


def get_data_loaders(dataset_path, train_batch_size, val_batch_size, tokenizer):
    dataloaders = {}
    dataloaders['train'] = DataLoader(ELI5(path.join(dataset_path,'q.tsv'), path.join(dataset_path, 'docs.tsv'), path.join(dataset_path, 'scores.json'), tokenizer),
        batch_size=train_batch_size, collate_fn=collate_fn_padd)
    # dataloaders['test'] =  DataLoader(ELI5(dataset_path+'test.p'), batch_size=val_batch_size, collate_fn=collate_fn_padd)
    # dataloaders['val'] =   DataLoader(ELI5(dataset_path+'val.p'), batch_size=val_batch_size, collate_fn=collate_fn_padd)

    return dataloaders
