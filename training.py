import torch
from utils import l1_loss, l0_loss
import numpy as np

def train(model, dataloader, optim, loss_fn, epochs, writer, log_interval=1, l1_reg_sparse_scalar=0.000001):
    loss_logs = list()
    for epoch in range(1, epochs+1):

        batch_loss= list()
        prev_model = model

        iter = 0

        for data, targets, lengths in dataloader:

            iter += 1
            logits = model(data, lengths)
            split_size = logits.size(0)//3
            q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

            l0_q, l0_docs = l0_loss(q_repr, d1_repr, d2_repr)

            loss = loss_fn(d1_repr, d2_repr, q_repr)

            aux_loss = l1_loss(q_repr, d1_repr, d2_repr)
            total_loss = loss + l1_reg_sparse_scalar * aux_loss
            optim.zero_grad()
            total_loss.backward()
            optim.step()

            batch_loss.append(loss.item())

            iter = (iter - 1) % len(dataloader) + 1
            if iter % log_interval == 0:
                print("Training - Epoch [{}] {}/{} loss: {:.4f}".format(epoch, iter, len(dataloader), loss.item()))

            writer.add_scalar('Loss', loss.item(), iter)
            writer.add_scalar('Aux loss', aux_loss.item(), iter)
            writer.add_scalar('L0 query', l0_q.item(), iter)
            writer.add_scalar('L0 docs', l0_docs.item(), iter)

    return model
