import pickle as p
import argparse
import random
import numpy as np
import os
def split(args):

	def gen_folds(dataset_len, num_folds):
		folds = list()
		rand_indices = list(range(dataset_len))
		random.shuffle(rand_indices)
		for i in range(1,num_folds+1):
			# train the model
			from_ = dataset_len*(i-1)//num_folds
			to_ = int(np.floor(dataset_len*i/num_folds))
			test_indices = rand_indices[from_:to_]
			train_indices = rand_indices[:from_] + rand_indices[to_:]
			folds.append([train_indices, test_indices])
		return folds
	
	
	q2idx = {}
	with open(args.triples_file, 'r') as in_f:
		count = 0
		for line in in_f:
			spl = line.split('\t')
			q_id = spl[0]
			if q_id not in q2idx:
				q2idx[q_id] = count
				count += 1
		base = '/'.join(args.triples_file.split('/')[:-1])
		out_folder = base + '/robust04_strong_supervision_folds.p'
		folds = gen_folds(len(q2idx), args.num_folds)
		p.dump(folds, open(out_folder, 'wb'))
		for i, (indices_train, indices_test) in enumerate(folds):
			out_fname = args.triples_file + '_'+ str(i)
			print(out_fname)
			with open(out_fname, 'w') as out_f:
				in_f.seek(0)
				for line in in_f:
					spl = line.split('\t')
					q_id = spl[0]
					if q2idx[q_id] in indices_train:
						out_f.write(line)
			 	
			os.system(f'sort -R -o {out_fname} {out_fname}')
				
if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--triples_file', type=str)
	parser.add_argument('--num_folds', type=int, default=5)
	args = parser.parse_args()
	split(args)

