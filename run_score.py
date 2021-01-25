from __future__ import print_function
import os.path
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from enlp.file_interface import File
from enlp.utils import load_embedding
from enlp.metrics import MAPTrec

from score.score_model import ScoreModel
from score.data_reader import DataReader





MODEL='score'
DEVICE = torch.device("cuda:0")  # torch.device("cpu"), if you want to run on CPU instead
MAX_QUERY_TERMS = 200
MAX_DOC_TERMS = 1500
NUM_HIDDEN_NODES = 300
VOCAB_SIZE = 0
MB_SIZE = 256
EPOCH_SIZE = 150
NUM_EPOCHS = 30
LEARNING_RATE = 0.0001
test_every = 1

DATA_DIR = 'data'
MODEL_DIR = 'experiments_duet'
idf = 'data/embeddings/glove.6B.300d.txt.vocab_robust04_idf_norm_weights.p'
DATA_FILE_VOCAB = os.path.join(DATA_DIR, "embeddings/word-vocab-small.tsv")
DATA_EMBEDDINGS = os.path.join(DATA_DIR, "embeddings/glove.6B.{}d.txt".format(NUM_HIDDEN_NODES))
DATA_FILE_IDFS = os.path.join(DATA_DIR, "embeddings/idfnew.norm.tsv")
DATA_FILE_TRAIN = os.path.join(DATA_DIR, "robust04/qrels.robust2004.txt_paper_binary_rand_docs_0")
DATA_FILE_DEV = os.path.join(DATA_DIR, "robust04/robust04_anserini_TREC_test_top_2000_bm25_fold_test_0")
QRELS_DEV = os.path.join(DATA_DIR, "robust04/qrels.robust2004.txt")
MODEL_FILE = os.path.join(MODEL_DIR, "duet.ep{}.dnn")


id2q = File('data/robust04/trec45-t.tsv_glove_stop_lucene_remove_unk.tsv')
id2d = File('data/robust04/robust04_raw_docs.num_query_glove_stop_lucene_remove_unk.tsv')

writer = SummaryWriter(MODEL_DIR)
net = ScoreModel([512, 512],load_embedding('data/embeddings/glove.6B.300d.p'), 300, 400002, 0.2, idf, trainable_weights=True)

READER_DEV = DataReader(DATA_FILE_DEV, 1, False, id2q, id2d, VOCAB_SIZE, NUM_HIDDEN_NODES, MAX_DOC_TERMS, MAX_QUERY_TERMS, DATA_FILE_VOCAB, DATA_EMBEDDINGS, MB_SIZE)
READER_TRAIN = DataReader(DATA_FILE_TRAIN, 2, True, id2q, id2d, VOCAB_SIZE, NUM_HIDDEN_NODES, MAX_DOC_TERMS, MAX_QUERY_TERMS, DATA_FILE_VOCAB, DATA_EMBEDDINGS, MB_SIZE)


def print_message(s):
	print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


print_message('Starting')
print_message('Learning rate: {}'.format(LEARNING_RATE))

net = net.to(DEVICE)
#criterion = nn.CrossEntropyLoss()
criterion = nn.MarginRankingLoss(margin=1)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)


for ep_idx in range(NUM_EPOCHS):
	# TRAINING
	net.train()
	train_loss = 0.0
	for mb_idx in range(EPOCH_SIZE):
		print(f'MB {mb_idx + 1}/{EPOCH_SIZE}')
		features = READER_TRAIN.get_minibatch()


		out = tuple([net(torch.from_numpy(features['q'][i]).to(DEVICE),
						torch.from_numpy(features['d'][i]).to(DEVICE),
						torch.from_numpy(features['lengths_q'][i]).to(DEVICE),
						torch.from_numpy(features['lengths_d'][i]).to(DEVICE)
					) for i in range(READER_TRAIN.num_docs)])

		#out = torch.cat(out, 1)
		#loss = criterion(out, torch.from_numpy(features['labels']).to(DEVICE))
		loss = criterion(out[0], out[1], torch.from_numpy(features['labels']).to(DEVICE))
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		train_loss += loss.item()
		print_message('epoch:{}, loss:{}'.format(ep_idx + 1, loss))
		writer.add_scalar('Loss/train batch', loss, ep_idx)

	print_message('epoch:{}, loss:{}'.format(ep_idx + 1, train_loss / (EPOCH_SIZE+1) ))
	writer.add_scalar('Train/loss', train_loss/ (EPOCH_SIZE+1), ep_idx)
	torch.save(net, MODEL_FILE.format(ep_idx + 1))

	# TESTING
	res_dev = {}
	original_scores = {}
	if ep_idx % test_every == 0:
		is_complete = False
		READER_DEV.reset()
		net.eval()
		while not is_complete:
			features = READER_DEV.get_minibatch()

			out = net(torch.from_numpy(features['q'][0]).to(DEVICE),
							torch.from_numpy(features['d'][0]).to(DEVICE),
							torch.from_numpy(features['lengths_q'][0]).to(DEVICE),
							torch.from_numpy(features['lengths_d'][0]).to(DEVICE))

			meta_cnt = len(features['meta'])
			out = out.data.cpu()
			for i in range(meta_cnt):
				q = features['meta'][i][0]
				d = features['meta'][i][1]
				orig_score = features['meta'][i][2]
				if q not in res_dev:
					res_dev[q] = {}
					original_scores[q] = {}
				if d not in res_dev[q]:
					res_dev[q][d] = -10000
					original_scores[q][d] = -10000
				original_scores[q][d] = orig_score
				res_dev[q][d] += out[i][0].detach().numpy()

			is_complete = (meta_cnt < MB_SIZE)

		sorted_scores = []
		sorted_scores_original = []
		q_ids = []

		for qid, docs in res_dev.items():
			sorted_scores_q = [(doc_id, docs[doc_id]) for doc_id in sorted(docs, key=docs.get, reverse=True)]
			sorted_scores_original_q = [(doc_id, original_scores[qid][doc_id]) for doc_id in sorted(original_scores[qid], key=original_scores[qid].get, reverse=True)]
			q_ids.append(qid)
			sorted_scores.append(sorted_scores_q)
			sorted_scores_original.append(sorted_scores_original_q)


		# RUN TREC_EVAL
		test = MAPTrec('trec_eval', 'data/robust04/qrels.robust2004.txt', 1000, ranking_file_path=f'{MODEL_DIR}/ranking')
		map_1000 = test.score(sorted_scores, q_ids)

		map_1000_original = test.score(sorted_scores_original, q_ids)
		print_message('original ranking model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		print_message('model:{}, map@1000:{}'.format(ep_idx + 1, map_1000))
		writer.add_scalar('Test/map@1000', map_1000, ep_idx)
