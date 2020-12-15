import torch
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import sys
import numpy as np
import pickle
def linear_cka(X,Y):

    def double_centering(gram):
        means = torch.mean(gram, 0)
        means -= torch.mean(means) / 2
        print(means.shape)
        gram -= means[:, None]
        gram -= means[None, :] 
        return gram
    def centering(gram):
        mean = gram.mean(0, keepdim=True)
        gram -= mean
        return gram
    def centering(K):
        n = K.shape[0]
        unit = np.ones([n, n])
        I = np.eye(n)
        H = I - unit / n

        return np.dot(np.dot(H, K), H) 

    X = centering(X)
    Y = centering(Y)
    XtX_F = torch.norm(torch.mm(X.T, X), p='fro')
    YtY_F = torch.norm(torch.mm(Y.T, Y), p='fro')
    YtX_F = torch.norm(torch.mm(Y.T, X), p='fro')

    return YtX_F**2 / (XtX_F*YtY_F)
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


def load(path):

	data = pickle.load(open(path, 'rb'))	
#	for f in glob.glob(f'{path}*.p'):
#		q_attention = pickle.load(open(f, 'rb'))
#		for  i, k in enumerate(q_attention):
#			for batch in q_attention[k]:
#				attentions[i].append(batch[:,:,:max_words])
	return data



def compare(A, sim_fn, B=None, layers_a=None, layers_b=None):
	if not layers_a:
		layers_a = range(len(A))
	if not B:
		B = A
	if not layers_b:
		layers_b = range(len(B))
	m = torch.zeros((len(layers_a), len(layers_b)))
	for i, a in enumerate(layers_a):
		for j, b in enumerate(layers_b):
			av_sim = list()
			assert len(A[a]) == len(B[b]), 'A and B are required to have the same number of batches'
			for batch in range(len(A[a])):
				if A[a][batch].shape[0] < 2:
					continue
				X = A[a][batch].reshape(A[a][batch].shape[0], -1)
				Y = B[b][batch].reshape(B[b][batch].shape[0], -1)
				# sanity check
				#Y[Y==0] = np.random.rand(Y.shape[0], Y.shape[1])[Y==0] * 0.01
				#X[X==0] = np.random.rand(X.shape[0], X.shape[1])[X==0] * 0.01
				sim = sim_fn(X,Y)
				av_sim.append(sim)
			m[i,j] = np.mean(av_sim) 
	return m


data = sys.argv[1]
attention = load(data)
m = compare(attention, sim_fn=linear_CKA)
sns.heatmap(pd.DataFrame(np.rot90(m, k=3), index=sorted(range(m.shape[0]), reverse=True), columns=range(m.shape[1])))
plt.savefig('name')
