from embedding_weighted_average import EmbeddingWeightedAverage
import torch


from torch import nn
from torch.autograd import Variable



if __name__ == "__main__":


	model = EmbeddingWeightedAverage('uniform', 10)

	x = torch.LongTensor([[1,2,9], [3,4,0]])
	lengths = torch.LongTensor([[3],[2]])
	values = torch.rand((2,3,10))
	y = model(x, values=values, lengths=lengths)
	print(y)
	target = Variable(torch.rand((2,10)))
	print(target)
	print(y)

	learning_rate = 1e-1
	optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
	loss_fn = torch.nn.MSELoss(size_average=False)

for i in range(100):
	optimizer.zero_grad()	
	x = torch.LongTensor([[1,2,9], [3,4,0]])
	lengths = torch.LongTensor([[3],[2]])
	values = torch.rand((2,3,10))
	y = model(x, values=values, lengths=lengths)
	loss = loss_fn(y, target)
	loss.backward()
	optimizer.step()
	print(model.weights.weight)
	print(loss)


