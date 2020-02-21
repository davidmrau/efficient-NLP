import torch
from utils import l1_loss, l0_loss
import numpy as np

def train(model, dataloaders, optim, loss_fn, epochs, writer, log_interval=1, l1_reg_sparse_scalar=0.000001, patience = 5):

    best_eval_loss = 1e20

    av_train_eval_loss = 0
    av_train_aux_loss = 0
    av_train_l0_q = 0
    av_train_l0_docs = 0
    av_train_task_loss = 0

    temp_patience = 0

    for epoch in range(1, epochs+1):

        batch_loss= list()
    
        iter = 0

        for data, targets, lengths in dataloaders["train"]:

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

            iter += 1
            if iter % log_interval == 0:
                print("Training - Epoch [{}] {}/{} loss: {:.4f}".format(epoch, iter, len(dataloaders["train"]), loss.item()))

            # sum losses
            av_train_loss += total_loss
            av_train_aux_loss += aux_loss
            av_train_l0_q += l0_q
            av_train_l0_docs += l0_docs
            av_train_task_loss += loss

            # update tensorboard
            writer.add_scalar('Train Task Loss', loss.item(), iter)
            writer.add_scalar('Train Aux loss', aux_loss.item(), iter)
            writer.add_scalar('Train Total Loss', total_loss.item(), iter)
            writer.add_scalar('Train L0 query', l0_q.item(), iter)
            writer.add_scalar('Train L0 docs', l0_docs.item(), iter)

        # average training losses
        av_train_loss /= len(dataloaders["train"])
        av_train_aux_loss /= len(dataloaders["train"])
        av_train_l0_q /= len(dataloaders["train"])
        av_train_l0_docs /= len(dataloaders["train"])
        av_train_task_loss /= len(dataloaders["train"])


        av_eval_loss, av_aux_loss, av_l0_q, av_l0_docs, av_task_loss = eval(model, dataloaders["eval"], loss_fn)

        print("Training   losses - Epoch [{}]: Total loss: {:.4f}, Task loss: {:.4f}, Aux loss: {:.4f}, Query l_0 : {:.4f}, Doc l_0: {:.4f}".format(epoch, av_train_loss, av_train_aux_loss, av_train_l0_q, av_train_l0_docs, av_train_task_loss))
        print("Evaluation losses - Epoch [{}]: Total loss: {:.4f}, Task loss: {:.4f}, Aux loss: {:.4f}, Query l_0 : {:.4f}, Doc l_0: {:.4f}".format(epoch, av_eval_loss, av_aux_loss, av_l0_q, av_l0_docs, av_task_loss))

        writer.add_scalar('Eval Task Loss', av_task_loss, iter)
        writer.add_scalar('Eval Aux loss', av_aux_loss, iter)
        writer.add_scalar('Eval Total Loss', av_eval_loss, iter)
        writer.add_scalar('Eval L0 query', av_l0_q, iter)
        writer.add_scalar('Eval L0 docs', av_l0_docs, iter)

        # check for early stopping
        if av_eval_loss < best_eval_loss:
            temp_patience = 0
            # save best model so far to file
            torch.save(model.state_dict(), "Best_model.model" )
        else:
            temp_patience += 1
            if temp_patience == patience:
                break

    # load best model
    model.load_state_dict(torch.load("Best_model.model"))

    return model


def eval(model, dataloader, loss_fn):

    av_eval_loss = 0
    av_aux_loss = 0
    av_l0_q = 0
    av_l0_docs = 0
    av_task_loss = 0

    for data, targets, lengths in dataloader:

        iter += 1
        logits = model(data, lengths)
        split_size = logits.size(0)//3
        q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

        l0_q, l0_docs = l0_loss(q_repr, d1_repr, d2_repr)

        av_l0_q += l0_q
        av_l0_docs += l0_docs

        loss = loss_fn(d1_repr, d2_repr, q_repr)

        av_task_loss += loss

        av_aux_loss += l1_loss(q_repr, d1_repr, d2_repr)
        av_eval_loss += loss + l1_reg_sparse_scalar * aux_loss


    av_eval_loss /= iter
    av_aux_loss /= iter
    av_l0_q /= iter
    av_l0_docs /= iter
    av_task_loss /= iter


    return av_eval_loss.item(), av_aux_loss.item(), av_l0_q.item(), av_l0_docs.item(), av_task_loss.item()