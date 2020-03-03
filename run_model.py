import torch
from utils import l1_loss, l0_loss
import numpy as np
import os

def run_epoch(model, dataloader, loss_fn, epoch, writer, l1_scalar, total_training_steps, device, optim=None):
    """Train 1 epoch, and evaluate every 1000 total_training_steps. Tensorboard is updated after every batch

    Returns
    -------
    av_loss                 : (float) average loss
    total_training_steps    : (int) the total number of training steps performed over all epochs
    type
        Description of returned object.
    """

    mode = 'Training' if optim != None else 'Dev'
    # initialize counters and sum variables
    av_loss, av_aux_loss, av_l0_q, av_l0_docs, av_task_loss, av_acc = 0, 0, 0, 0, 0, 0

    num_batches = len(dataloader)
    training_steps = 0
    for data, targets, lengths in dataloader:
        total_training_steps += 1
        training_steps += 1
        targets = targets.to(device)
        # forward pass (inputs are concatenated in the form [q1, q2, ..., q1d1, q2d1, ..., q1d2, q2d2, ...])
        logits = model(data.to(device), lengths.to(device))
        split_size = logits.size(0)//3
        # accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
        q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

        # performing inner products
        dot_q_d1 = torch.bmm(q_repr.unsqueeze(1), d1_repr.unsqueeze(-1)).squeeze()
        dot_q_d2 = torch.bmm(q_repr.unsqueeze(1), d2_repr.unsqueeze(-1)).squeeze()

        # calculating loss
        loss = loss_fn(dot_q_d1, dot_q_d2, targets.to(device))
        aux_loss = l1_loss(q_repr, d1_repr, d2_repr) * l1_scalar
        # calculating L0 loss
        l0_q, l0_docs = l0_loss(d1_repr, d2_repr, q_repr)
        # calculating classification accuracy (whether the correct document was classified as more relevant)
        acc = (((dot_q_d1 > dot_q_d2).float() == targets).float()+ ((dot_q_d2 > dot_q_d1).float() == targets*-1).float()).mean()

        # aggregating losses and running backward pass and update step
        total_loss = loss +  aux_loss
        if optim != None:
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        # update tensorboard every 1000 training steps
        freq = num_batches // 10000 if num_batches > 10000 else 10
        #freq = 100
        if training_steps % freq == 0:
            print("  {}/{} task loss: {:.4f}, aux loss: {:.4f}".format(training_steps, num_batches, loss.item(), aux_loss.item()))
            # update tensorboard
            writer.add_scalar(f'{mode}_task_loss', loss.item(), total_training_steps  )
            writer.add_scalar(f'{mode}_aux_loss', aux_loss.item(), total_training_steps)
            writer.add_scalar(f'{mode}_total_loss', total_loss.item(), total_training_steps)
            writer.add_scalar(f'{mode}_L0_query', l0_q, total_training_steps)
            writer.add_scalar(f'{mode}_L0_docs', l0_docs, total_training_steps)
            writer.add_scalar(f'{mode}_acc', acc.item(), total_training_steps)

        # sum losses
        av_loss += total_loss.item()
        av_aux_loss += aux_loss.item()
        av_l0_q += l0_q
        av_l0_docs += l0_docs
        av_task_loss += loss.item()

        # calculate av_acc
        av_acc += acc


        if optim == None:
            # eval
            if training_steps > 1000 :
                break

        if training_steps > 10000 :
            break
    # average losses and counters
    av_loss /= training_steps
    av_aux_loss /= training_steps
    av_l0_q /= training_steps
    av_l0_docs /= training_steps
    av_task_loss /= training_steps
    av_acc /= training_steps


    print("{} - Epoch [{}]: Total loss: {:.6f}, Task loss: {:.6f}, Aux loss: {:.6f}, Query l_0 : {:.4f}, Doc l_0: {:.4f}, acc: {:.4f}".format(mode, epoch, av_loss ,av_task_loss, av_aux_loss, av_l0_q, av_l0_docs, av_acc))

    return av_loss, total_training_steps

def train(model, dataloaders, optim, loss_fn, epochs, writer, device, l1_scalar = None, patience = 5):
    """Takes care of the complete training procedure (over epochs, while evaluating)

    Parameters
    ----------
    model       : Pytorch model (Randomly initialized)
    dataloaders : dataloaders
    optim       : Pytorch optimizer
    loss_fn     : Task loss (MarginRankingLoss)
    epochs      : int, Max training epochs
    writer      : Tensorboard writter
    device      : (CPU or CUDA, defined in main.py)
    l1_scalar   : float
        L1 loss multiplier, affecting the total loss
    patience    : int
        Training patience

    Returns
    -------
    type
        Best model found throughout the training

    """

    best_eval_loss = 1e20
    temp_patience = 0
    total_training_steps = 0

    for epoch in range(1, epochs+1):
        # training
        with torch.enable_grad():
            model.train()
            av_train_loss, total_training_steps = run_epoch(model, dataloaders['train'], loss_fn, epoch, writer, l1_scalar, total_training_steps, device,  optim=optim)
        # evaluation
        with torch.no_grad():
            model.eval()
            av_eval_loss, _ = run_epoch(model, dataloaders['val'], loss_fn, epoch, writer, l1_scalar, total_training_steps, device)

        # check for early stopping
        if av_eval_loss < best_eval_loss:
            print(f'Best model at current epoch {epoch}')
            temp_patience = 0
            av_eval_loss = best_eval_loss
            # save best model so far to file
            torch.save(model, f'best_model.model' )
        else:
            temp_patience += 1
            if temp_patience == patience:
                break

        torch.save(model, f'model_epoch_{epoch}.model' )
    # load best model
    torch.load(f'best_model.model')

    return model
