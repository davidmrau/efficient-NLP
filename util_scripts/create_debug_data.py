import pickle

import hydra
from hydra import utils
from utils import *


@hydra.main(config_path='config.yaml')
def exp(cfg):

    orig_cwd = utils.get_original_cwd() + '/'
    dataset_path = orig_cwd + cfg.dataset_path
    triplets = read_triplets(f'{dataset_path}/qidpidtriples.train.full.debug.tsv')
    docs = read_pickle(f'{dataset_path}/collection.tsv.p')
    qrels = read_qrels(f'{dataset_path}/qrels.dev.tsv')
    debug_docs = {}

    for _, d1_id,d2_id in triplets:
        debug_docs[d1_id] = docs[d1_id]
        debug_docs[d2_id] = docs[d2_id]
    for _, d in qrels:
        debug_docs[d] = docs[d]
    pickle.dump(debug_docs, open(f'{dataset_path}/collection.tsv.debug.p', 'wb'))

if __name__ == "__main__":
    exp()
