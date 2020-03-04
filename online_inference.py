from dataset import MSMarcoSequential
import torch

from inverted_index import InvertedIndex
from torch.nn.utils.rnn import pad_sequence
from omegaconf import OmegaConf

""" Load an pre-built inverted index and run online inference (for test set)
"""
# usage: online_inference.py model_folder=FOLDER_TO_MODEL
# the script is loading FOLDER_TO_MODEL/best_model.model
#



def exp(cfg):

    if not cfg.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    split = cfg.split


    model = torch.load(cfg.model_path)

	# Initialize an Inverted Index object
	ii = InvertedIndex(path=cfg.model_folder, vocab_size = cfg.sparse_dimensions, num_of_workers=cfg.num_of_workers_index)

	filename = f'{cfg.dataset_path}/queries.{split}.tokenized.tsv'

	ms_batch_generator = MSMarcoSequential(filename, cfg.batch_size).batch_generator()

	model = torch.load(cfg.model_folder + '/best_model.model', map_location=device)

    # open results file
    open(os.path.join(cfg.model_folder + '/ranking_results.' + split , 'w').close()

	for batch_ids, batch_data, batch_lengths in ms_batch_generator:
		# print(batch_data)
		logits = model(batch_data.to(device), batch_lengths.to(device))
        results = ii.get_scores(ids.cpu().numpy(), logits.cpu(), top_results = 10, max_candidates_per_posting_list = -1)
        # write them to the file in the form:
        # q_id top_1_doc_id rank
        # ...
        # 1124703 8766037 1
        # 1124703 8021997 2
        # 1124703 7816201 3
        print(results)
        exit()


if __name__ == "__main__":
    # getting command line arguments
    cl_cfg = OmegaConf.from_cli()
    # getting model config
    if not cl_cfg.model_folder:
        raise ValueError("usage: online_inference.py model_folder=FOLDER_TO_MODEL")
    # getting model config
    if not cl_cfg.split or cl_cfg.split not in ['train', 'dev', 'eval']:
        raise ValueError("usage: specify the split :{train/dev/eval}")
    cfg_load = OmegaConf.load(f'{cl_cfg.model_folder}/config.yaml')
    # merging both
    cfg = OmegaConf.merge(cfg_load, cl_cfg)
    exp(cfg)
