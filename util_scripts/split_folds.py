import pickle as p
import argparse
import random
import numpy as np
import os
import subprocess
def split(args):

	def gen_folds(dataset_len, num_folds, idx2q):
		folds = list()
		rand_indices = list(range(dataset_len))
		random.shuffle(rand_indices)
		for i in range(1,num_folds+1):
			# train the model
			from_ = dataset_len*(i-1)//num_folds
			to_ = int(np.floor(dataset_len*i/num_folds))
			test_indices = [ idx2q[l] for l in rand_indices[from_:to_]]
			train_indices = [idx2q[l] for l in  rand_indices[:from_] + rand_indices[to_:]]
			folds.append([train_indices, test_indices])
		return folds
	
	
	with open(args.triples_file, 'r') as in_f:
		count = 0
		idx2q = {}
		q2idx = {}
		for line in in_f:
			spl = line.split('\t')
			q_id = spl[0]
			if q_id not in q2idx:
				q2idx[q_id] = count
				idx2q[count] = q_id
				count += 1
		print(idx2q)
		if not args.folds:
			out_folder = args.triples_file + '.p'
			folds = gen_folds(len(idx2q), args.num_folds, idx2q)	
			p.dump(folds, open(out_folder, 'wb'))
		else:
			folds = p.load(open(args.folds, 'rb'))
		for i, (indices_train, indices_test) in enumerate(folds):
			#exit()
			#print(sorted(indices_train), sorted(indices_test))
			out_fname = args.triples_file + '_'+ str(i)
			#print(out_fname)
			with open(out_fname, 'w') as out_f:
				in_f.seek(0)
				for line in in_f:
					spl = line.split('\t')
					q_id = spl[0]
					if q_id in indices_train:
						out_f.write(line)
					else:
						print(q_id)
			subprocess.run(['shuf', out_fname, '-o', out_fname])
			print('done shuffling', out_fname)
if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--triples_file', type=str)
	parser.add_argument('--folds', default=None, type=str)
	parser.add_argument('--num_folds', type=int, default=5)
	args = parser.parse_args()
	split(args)

