import torch
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import sys
import numpy as np
import pickle

import argparse


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    H = I - unit / n

    return np.dot(np.dot(H, K), H)  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH

def linear_HSIC(X, Y):
    L_X = np.dot(X, X.T)
    L_Y = np.dot(Y, Y.T)
    return np.sum(centering(L_X) * centering(L_Y))


def linear_CKA(X, Y):
    hsic = linear_HSIC(X, Y)
    var1 = np.sqrt(linear_HSIC(X, X))
    var2 = np.sqrt(linear_HSIC(Y, Y))

    return hsic / (var1 * var2)

def linear_cka(X,Y):
	def cen(X):
		means = X.mean(0, keepdims=True)
		return X - means
	X = cen(X)
	Y = cen(Y)
	XX = np.linalg.norm(X.T @ X)
	YX = np.linalg.norm(Y.T @ X)
	YY = np.linalg.norm(Y.T @ Y)
	sim = YX**2 / (XX*YY)

	return sim

def load(path):

	data = pickle.load(open(path, 'rb'))	
#	for f in glob.glob(f'{path}*.p'):
#		q_attention = pickle.load(open(f, 'rb'))
#		for  i, k in enumerate(q_attention):
#			for batch in q_attention[k]:
#				attentions[i].append(batch[:,:,:max_words])
	return data



def compare(A, B, sim_fn, layers_a=None, layers_b=None):

	if not layers_a:
		layers_a = range(len(A))
	if not layers_b:
		layers_b = range(len(B))

	m = torch.zeros((len(layers_a), len(layers_b)))
	print(m.shape)
	for i, a in enumerate(layers_a):
		for j, b in enumerate(layers_b):
			av_sim = list()
			assert len(A[a]) == len(B[b]), f'A and B are required to have the same number of batches. A: {len(A[a])}, B: {len(B[b])}'
			for batch in range(len(A[a])):
				if A[a][batch].shape[0] < 2:
					continue
				X = A[a][batch].reshape(-1, A[a][batch].shape[0])
				Y = B[b][batch].reshape(-1, B[b][batch].shape[0])
				print(X.shape, Y.shape)
				# sanity check
				#Y[Y==0] = np.random.rand(Y.shape[0], Y.shape[1])[Y==0] * 0.01
				#X[X==0] = np.random.rand(X.shape[0], X.shape[1])[X==0] * 0.01
				sim = sim_fn(X,Y)
				av_sim.append(sim)
			m[i,j] = np.mean(av_sim) 
	return m


def main(args):
	data_a = load(args.file_a)
	if args.file_b:
		data_b = load(args.file_b)
	else:
		data_b = data_a

	if args.sim_fn == 'linear_cka':
		sim_fn=linear_cka
	else:
		raise ValueError(f'similarity {args.sim_fun} doesn\'t exist!')
	m = compare(data_a, data_b, sim_fn)
	print(m)
	m = np.rot90(m, k=3)
	sns.heatmap(pd.DataFrame(m, index=sorted(range(m.shape[0]), reverse=True), columns=range(m.shape[1])), vmin=0, vmax=1, annot=True)
	plt.tight_layout()
	if not args.file_b:
		plt.savefig(f'{args.file_a}_heat.pdf')
	else:
		base_file_b = args.file_b.split('/')[-4]
		plt.savefig(f'{args.file_a}_{base_file_b}_heat.pdf')


if __name__ == '__main__':


	parser = argparse.ArgumentParser(description='Representation similarity. Provide files containing representations in the form: (layers x batch x ...dims')
	parser.add_argument('--file_a', type=str, required=True, help='If only file_a is provided compare layers of file_a')
	parser.add_argument('--file_b', type=str, default=None, help='If provided compare with file_a')
	parser.add_argument('--sim_fn', type=str, default='linear_cka', help='Similarity function that is applied.')
	args = parser.parse_args()
	main(args)
