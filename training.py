import torch
from utils import l1_loss, l0_loss
import numpy as np
import os

def run_epoch(model, dataloader, loss_fn, epoch, writer, l1_scalar, steps, device, optim=None):

    mode = 'Training' if optim != None else 'Test'
    av_loss, av_aux_loss, av_l0_q, av_l0_docs, av_task_loss = 0, 0, 0, 0, 0

    num_batches = len(dataloader)
    steps_epoch = 0
    for data, targets, lengths in dataloader:
        steps += 1
        steps_epoch += 1

        logits = model(data.to(device), lengths)

        split_size = logits.size(0)//3
        q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

        dot_q_d1 = q_repr @ d1_repr.T
        dot_q_d2 = q_repr @ d2_repr.T

        loss = loss_fn(dot_q_d1, dot_q_d2, targets)

        aux_loss = l1_loss(q_repr, d1_repr, d2_repr) * l1_scalar
        l0_q, l0_docs = l0_loss(d1_repr, d2_repr, q_repr)

        total_loss = loss +  aux_loss
        if optim is not None:
            optim.zero_grad()
            total_loss.backward()
            optim.step()

        if steps_epoch % num_batches // 25 == 0 and steps_epoch != 0:
            print("  {}/{} task loss: {:.4f}, aux loss: {:.4f}".format(steps_epoch, num_batches, loss.item(), aux_loss.item()))
            # update tensorboard
            writer.add_scalar(f'{mode}_task_loss', loss.item(), steps  )
            writer.add_scalar(f'{mode}_aux_loss', aux_loss.item(), steps)
            writer.add_scalar(f'{mode}_total_oss', total_loss.item(), steps)
            writer.add_scalar(f'{mode}_L0_query', l0_q, steps)
            writer.add_scalar(f'{mode}_L0_docs', l0_docs, steps)

        # sum losses
        av_loss += total_loss.item()
        av_aux_loss += aux_loss.item()
        av_l0_q += l0_q
        av_l0_docs += l0_docs
        av_task_loss += loss.item()


    # average training losses
    av_loss /= num_batches
    av_aux_loss /= num_batches
    av_l0_q /= num_batches
    av_l0_docs /= num_batches
    av_task_loss /= num_batches


    print("{} - Epoch [{}]: Total loss: {:.4f}, Task loss: {:.4f}, Aux loss: {:.4f}, Query l_0 : {:.4f}, Doc l_0: {:.4f}".format(mode, epoch, av_loss ,av_task_loss, av_aux_loss, av_l0_q, av_l0_docs,))

    return av_loss, steps

def run(model, dataloaders, optim, loss_fn, epochs, writer, l1_scalar = None, patience = 5):

    best_eval_loss = 1e20
    temp_patience = 0
    steps = 0

    for epoch in range(1, epochs+1):
        # training
        with torch.enable_grad():
            model.train()
            av_train_loss, steps = run_epoch(model, dataloaders['train'], loss_fn, epoch, writer, l1_scalar, steps, optim=optim)
        # evaluation
        with torch.no_grad():
            model.eval()
            av_eval_loss, steps = run_epoch(model, dataloaders['test'], loss_fn, epoch, writer, l1_scalar, steps)

        # check for early stopping
        if av_eval_loss < best_eval_loss:
            temp_patience = 0
            av_eval_loss = best_eval_loss
            # save best model so far to file
            torch.save(model.state_dict(), f'{os.getcwd()}/best_model.model' )
        else:
            temp_patience += 1
            if temp_patience == patience:
                break

        torch.save(model.state_dict(), f'{os.getcwd()}/model_epoch_{epoch}.model' )
    # load best model
    model.load_state_dict(torch.load(f'{os.getcwd()}/best_model.model'))

    return model
