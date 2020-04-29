import torch
from utils import l1_loss_fn, l0_loss_fn, balance_loss_fn, write_ranking, l0_loss
import numpy as np
import os
from inference import evaluate
from ms_marco_eval import compute_metrics_from_files
from utils import plot_histogram_of_latent_terms, plot_ordered_posting_lists_lengths

def run_epoch(model, dataloader, loss_fn, epoch, writer, l1_scalar, balance_scalar, total_training_steps, device, optim=None, eval_every = 10000):
	"""Train 1 epoch, and evaluate every 1000 total_training_steps. Tensorboard is updated after every batch

	Returns
	-------
	av_loss                 : (float) average loss
	total_training_steps    : (int) the total number of training steps performed over all epochs
	type
		Description of returned object.
	"""

	mode = 'train' if optim != None else 'val'
	# initialize counters and sum variables
	av_loss, av_l1_loss, av_balance_loss, av_l0_q, av_l0_docs, av_task_loss, av_acc = 0, 0, 0, 0, 0, 0, 0

	num_batches = len(dataloader)
	training_steps = 0
	for batch in dataloader:

		if batch is None:
			continue

		data, targets, lengths = batch

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


		l1_loss = l1_loss_fn(torch.cat([q_repr, d1_repr, d2_repr], 1))

		balance_loss = balance_loss_fn(logits, device) * balance_scalar
		# calculating L0 loss
		l0_q, l0_docs = l0_loss_fn(d1_repr, d2_repr, q_repr)

		# calculating classification accuracy (whether the correct document was classified as more relevant)
		acc = (((dot_q_d1 > dot_q_d2).float() == targets).float()+ ((dot_q_d2 > dot_q_d1).float() == targets*-1).float()).mean()

		# aggregating losses and running backward pass and update step
		total_loss = loss +  l1_loss * l1_scalar + balance_loss
		if optim != None:
			optim.zero_grad()
			total_loss.backward()
			optim.step()

		torch.cuda.empty_cache()
		# calculate tensorboard update dynamically
		print_n_times = eval_every
		freq = num_batches // print_n_times if num_batches > print_n_times else 1
		# update tensorboard only for training on intermediate steps
		if training_steps % freq == 0 and mode == 'train':
			print("  {}/{} task loss: {:.4f}, l1 loss: {:.4f}, balance loss: {:.4f}".format(training_steps, num_batches, loss, l1_loss, balance_loss))
				# update tensorboard
			print("Train :loss", loss)
			writer.add_scalar(f'{mode}_task_loss', loss, total_training_steps  )
			writer.add_scalar(f'{mode}_l1_loss', l1_loss, total_training_steps)
			writer.add_scalar(f'{mode}_balance_loss', balance_loss, total_training_steps)
			writer.add_scalar(f'{mode}_total_loss', total_loss, total_training_steps)
			writer.add_scalar(f'{mode}_L0_query', l0_q, total_training_steps)
			writer.add_scalar(f'{mode}_L0_docs', l0_docs, total_training_steps)
			writer.add_scalar(f'{mode}_acc', acc, total_training_steps)


		# sum losses
		av_loss += total_loss
		av_l1_loss += l1_loss
		av_balance_loss += balance_loss
		av_l0_q += l0_q
		av_l0_docs += l0_docs
		av_task_loss += loss

		# calculate av_acc
		av_acc += acc

		if training_steps > eval_every:
			break

	# average losses and counters
	av_loss = av_loss / training_steps
	av_l1_loss = av_l1_loss /training_steps
	av_balance_loss = av_balance_loss / training_steps
	av_l0_q /= training_steps
	av_l0_docs /= training_steps
	av_task_loss = av_task_loss / training_steps
	av_acc /= training_steps
	av_acc = av_acc


	print("{} - Epoch [{}]: Total loss: {:.6f}, Task loss: {:.6f}, L1 loss: {:.6f}, Balance Loss: {:.6f}, Query l_0 : {:.4f}, Doc l_0: {:.4f}, acc: {:.4f}".format(mode, epoch, av_loss ,av_task_loss, av_l1_loss, av_balance_loss, av_l0_q, av_l0_docs, av_acc))

	# for validation only send average to tensorboard
	if mode == 'val':
		writer.add_scalar(f'{mode}_task_loss', loss, total_training_steps  )
		writer.add_scalar(f'{mode}_l1_loss', l1_loss, total_training_steps)
		writer.add_scalar(f'{mode}_balance_loss', balance_loss, total_training_steps)
		writer.add_scalar(f'{mode}_total_loss', total_loss, total_training_steps)
		writer.add_scalar(f'{mode}_L0_query', l0_q, total_training_steps)
		writer.add_scalar(f'{mode}_L0_docs', l0_docs, total_training_steps)
		writer.add_scalar(f'{mode}_acc', acc, total_training_steps)

	return av_loss, total_training_steps

def train(model, dataloaders, optim, loss_fn, epochs, writer, device, model_folder, qrels, dataset_path, sparse_dimensions, top_results, l1_scalar = 1, balance_scalar= 1, patience = 2, MaxMRRRank=1000, eval_every = 10000, debug = False, bottleneck_run = False):
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
	model_folder: str, folder to save the model to
	l1_scalar   : float
		L1 loss multiplier, affecting the total loss
	patience    : int
		Training patience

	Returns
	-------
	type
		Best model found throughout the training

	"""

	# best_eval_loss = 1e20
	best_MRR = -1
	temp_patience = 0
	total_training_steps = 0
	qrels_base = qrels.split('/')[-1]
	ranking_file_path = f'{model_folder}/{qrels_base}_full_ranking'
	tmp_ranking_file = f'{ranking_file_path}_tmp'

	for epoch in range(1, epochs+1):
		print('Epoch', epoch)
		# training
		with torch.enable_grad():
			model.train()
			av_train_loss, total_training_steps = run_epoch(model, dataloaders['train'], loss_fn, epoch, writer, l1_scalar, balance_scalar, total_training_steps, device,  optim=optim, eval_every = eval_every)
		# evaluation
		with torch.no_grad():
			model.eval()

			if bottleneck_run:
				MRR = 0.001

			else:
				# run ms marco eval
				scores, q_repr, d_repr, q_ids, _ = evaluate(model, dataloaders['val'], device, top_results)

				q_l0_loss = 0
				q_l0_coutner = 0
				l1_loss = 0
				l1_counter = 0
				# calculate average l1 and l0 losses from queries
				for i in range(len(q_repr)):
					number_of_samples = q_repr[i].size(0)
					l1_loss += l1_loss_fn(q_repr[i])
					q_l0_loss += l0_loss(q_repr[i])
					q_l0_coutner += number_of_samples
					l1_counter += number_of_samples

				d_l0_loss = 0
				d_l0_coutner = 0
				# calculate average l1 and l0 losses from documents
				for i in range(len(d_repr)):
					number_of_samples = d_repr[i].size(0)
					l1_loss += l1_loss_fn(d_repr[i])
					d_l0_loss += l0_loss(d_repr[i])
					d_l0_coutner += number_of_samples
					l1_counter += number_of_samples

				l1_loss /= l1_counter
				q_l0_loss /= q_l0_coutner
				d_l0_loss /= d_l0_coutner


				writer.add_scalar(f'Eval_l1_loss', l1_loss, total_training_steps)

				writer.add_scalar(f'Eval_L0_query', q_l0_loss, total_training_steps)
				writer.add_scalar(f'Eval_L0_docs', d_l0_loss, total_training_steps)



				
				write_ranking(scores, q_ids, tmp_ranking_file, MaxMRRRank)

				metrics = compute_metrics_from_files(path_to_reference = qrels, path_to_candidate = tmp_ranking_file, MaxMRRRank=MaxMRRRank)
				MRR = metrics[f'MRR @{MaxMRRRank}']


				writer.add_scalar(f'Eval_MRR@1000', MRR, total_training_steps  )
				print(f'Eval -  MRR@1000: {MRR}')


				# check for early stopping
				if MRR > best_MRR:
					print(f'Best model at current epoch {epoch}, with MRR@1000: {MRR}')
					temp_patience = 0
					best_MRR = MRR
					# save best model so far to file
					torch.save(model, f'{model_folder}/best_model.model' )

					# write ranking file
					write_ranking(scores, q_ids, ranking_file_path, MaxMRRRank)
					# plot stats
					plot_ordered_posting_lists_lengths(model_folder, q_repr, 'query')
					plot_histogram_of_latent_terms(model_folder, q_repr, sparse_dimensions, 'query')
					plot_ordered_posting_lists_lengths(model_folder, d_repr, 'docs')
					plot_histogram_of_latent_terms(model_folder, d_repr, sparse_dimensions, 'docs')

				else:

					temp_patience += 1

					if temp_patience >= patience:
						print("Early Stopping!")
						break

				if MRR < 0.05 and not debug:
					print("MRR smaller than 0.05. Ending Training!")
					break


	if not bottleneck_run:
		# load best model
		model = torch.load(f'{model_folder}/best_model.model')
		os.remove(tmp_ranking_file)
	


	return model
