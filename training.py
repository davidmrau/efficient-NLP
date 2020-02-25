import torch
from utils import l1_loss, l0_loss
import numpy as np


def run_epoch(model, dataloader, loss_fn, epoch, writer, l1_reg_sparse_scalar, steps, optim=None):

    mode = 'Training' if optim != None else 'Test'

    av_loss, av_aux_loss, av_l0_q, av_l0_docs, av_task_loss = 0, 0, 0, 0, 0

    num_batches = len(dataloader)

    for data, targets, lengths in dataloader:
        steps += 1
        logits = model(data, lengths)

        split_size = logits.size(0)//3
        q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

        dot_q_d1 = q_repr @ d1_repr.T
        dot_q_d2 = q_repr @ d2_repr.T

        loss = loss_fn(dot_q_d1, dot_q_d2, targets)

        aux_loss = l1_loss(q_repr, d1_repr, d2_repr) * l1_reg_sparse_scalar
        l0_q, l0_docs = l0_loss(d1_repr, d2_repr, q_repr)

        total_loss = loss +  aux_loss
        if optim is not None:
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        if iter % num_batches // 10 == 1:
            print("  {}/{} task loss: {:.4f}, aux loss: {:.4f}".format(iter, num_batches, av_task_loss/iter, av_aux_loss/iter))

            # update tensorboard
            writer.add_scalar(f'{mode} Task Loss', av_task_loss/iter, iter  )
            writer.add_scalar(f'{mode} Aux loss', av_aux_loss/iter, iter)
            writer.add_scalar(f'{mode} Total Loss', av_loss/iter, iter)
            writer.add_scalar(f'{mode} L0 query', av_l0_q/iter, iter)
            writer.add_scalar(f'{mode} L0 docs', av_l0_docs/iter, iter)

        # sum losses
        av_loss += total_loss.item()
        av_aux_loss += aux_loss.item()
        av_l0_q += l0_q
        av_l0_docs += l0_docs
        av_task_loss += loss.item()
        
        if iter >= 100:
            break

    # average training losses
    av_loss /= num_batches
    av_aux_loss /= num_batches
    av_l0_q /= num_batches
    av_l0_docs /= num_batches
    av_task_loss /= num_batches


    print("{} - Epoch [{}]: Total loss: {:.4f}, Task loss: {:.4f}, Aux loss: {:.4f}, Query l_0 : {:.4f}, Doc l_0: {:.4f}".format(mode, epoch, av_loss ,av_task_loss, av_aux_loss, av_l0_q, av_l0_docs,))

    return av_loss, steps

def run(model, dataloaders, optim, loss_fn, epochs, writer, l1_reg_sparse_scalar = 0.01, patience = 5):

    best_eval_loss = 1e20
    temp_patience = 0
    steps = 0

    for epoch in range(1, epochs+1):
        # training
        with torch.enable_grad():
            model.train()
            av_train_loss, steps = run_epoch(model, dataloaders['train'], loss_fn, epoch, writer, l1_reg_sparse_scalar, optim=optim)
        # evaluation
        with torch.no_grad():
            model.eval()
            av_eval_loss, steps = run_epoch(model, dataloaders['test'], loss_fn, epoch, writer, l1_reg_sparse_scalar, steps)

        # check for early stopping
        if av_eval_loss < best_eval_loss:
            temp_patience = 0
            av_eval_loss = best_eval_loss
            # save best model so far to file
            torch.save(model.state_dict(), "Best_model.model" )
        else:
            temp_patience += 1
            if temp_patience == patience:
                break

    # load best model
    model.load_state_dict(torch.load("Best_model.model"))

    return model
