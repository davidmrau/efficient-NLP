
from utils import get_data_loaders

from snrm import SNRM
dataset_path = ''
train_batch_size = 4
val_batch_size = 4
embedding_size = 32
hidden_sizes = [128, 64, 10000]
n = 5



def l1_reg(q_repr, d1_repr, d2_repr):
    return torch.mean(torch.sum(torch.cat([self.q_repr, self.d1_repr, self.d2_repr], dim=1), dim=1))



dataloaders = get_data_loaders(dataset_path, train_batch_size, val_batch_size)
snrm = SNRM(embedding_size=embedding_size, hidden_sizes=hidden_sizes, n=n)

for x,y, lengths in dataloaders['train']:
    print('x.shape', x.shape, lengths)
    repr = snrm(x, lengths)
    print(repr.shape)
    break
