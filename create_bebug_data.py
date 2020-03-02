from utils import *
import hydra
from hydra import utils
import os
import pickle


@hydra.main(config_path='config.yaml')
def exp(cfg):

    orig_cwd = utils.get_original_cwd() + '/'
    dataset_path = orig_cwd + cfg.dataset_path
    triplets = read_triplets(f'{dataset_path}/qidpidtriples.train.full.debug.tsv')
    docs = read_pickle(f'{dataset_path}/collection.tsv.p')

    debug_docs = {}

    for _, d1_id,d2_id in triplets:
        debug_docs[d1_id] = docs[d1_id]
        debug_docs[d2_id] = docs[d2_id]
    pickle.dump(debug_docs, open(f'{dataset_path}/collection.tsv.debug.p', 'wb'))

if __name__ == "__main__":
    exp()
