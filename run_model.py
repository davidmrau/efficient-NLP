import torch
import numpy as np
from utils import l1_loss_fn, l0_loss_fn, balance_loss_fn, l0_loss, plot_histogram_of_latent_terms, \
	plot_ordered_posting_lists_lengths, Average, EarlyStopping, split_batch_to_minibatches


def log_progress(mode, total_trained_samples, currently_trained_samples, samples_per_epoch, loss, l1_loss,
				 balance_loss, total_loss, l0_q, l0_docs, acc, writer=None):
	print("{}  {}/{} total loss: {:.4f}, task loss: {:.4f}, l1 loss: {:.4f}, balance loss: {:.4f}".format(mode, currently_trained_samples, samples_per_epoch, total_loss, loss, l1_loss, balance_loss))
	if writer:
		# update tensorboard
		writer.add_scalar(f'{mode}_task_loss', loss, total_trained_samples)
		writer.add_scalar(f'{mode}_l1_loss', l1_loss, total_trained_samples)
		writer.add_scalar(f'{mode}_balance_loss', balance_loss, total_trained_samples)
		writer.add_scalar(f'{mode}_total_loss', total_loss, total_trained_samples)
		writer.add_scalar(f'{mode}_L0_query', l0_q, total_trained_samples)
		writer.add_scalar(f'{mode}_L0_docs', l0_docs, total_trained_samples)
		writer.add_scalar(f'{mode}_acc', acc, total_trained_samples)


def run_epoch(model, mode, dataloader, batch_iterator, loss_fn, epoch, writer, l1_scalar, balance_scalar,
			  total_trained_samples, device, optim=None, samples_per_epoch=10000, log_every_ratio=0.01,
			  max_samples_per_gpu = 16, n_gpu = 1):
	"""Train 1 epoch, and evaluate every 1000 total_training_steps. Tensorboard is updated after every batch

	Returns
	-------
	av_loss                 : (float) average loss
	total_trained_samples    : (int) the total number of training steps performed over all epochs
	type
		Description of returned object.
	"""
	log_every_ratio = max(dataloader[mode].batch_size / samples_per_epoch, log_every_ratio)
	prev_trained_samples = total_trained_samples

	current_trained_samples = prev_trained_samples - total_trained_samples

	current_log_threshold = log_every_ratio

	cur_trained_samples = 0

	av_loss, av_l1_loss, av_balance_loss, av_total_loss, av_l0_q, av_l0_docs, av_acc = Average(), Average(), Average(), Average(), Average(), Average(), Average()

	while cur_trained_samples < samples_per_epoch:
		try:
			batch = next(batch_iterator)
		except StopIteration:
			# StopIteration is thrown if dataset ends
			# reinitialize data loader
			batch_iterator = iter(dataloader[mode])
			batch = next(batch_iterator)

		# For the weak supervision setting, some doc/queries are empty, rsulting to that sample being None
		# If all samples are None within a batch, then the batch is None
		if batch is None:
			continue

		if optim != None:
			optim.zero_grad()

		minibatches = split_batch_to_minibatches(batch, max_samples_per_gpu = max_samples_per_gpu, n_gpu=n_gpu)

		batch_samples_number = 0

		# allocate space in tensor form to sum the losses, per task

		for minibatch in minibatches:

			# decompose batch
			data, targets, lengths = minibatch
			# get number of samples within the batch
			minibatch_samples_number = targets.size(0)

			batch_samples_number += minibatch_samples_number

			# update the number of trained samples in this epoch
			cur_trained_samples += minibatch_samples_number
			# update the total number of trained samples
			total_trained_samples += minibatch_samples_number

			# forward pass (inputs are concatenated in the form [q1, q2, ..., q1d1, q2d1, ..., q1d2, q2d2, ...])
			logits = model(data.to(device), lengths.to(device))
			# moving targets also to the appropriate device
			targets = targets.to(device)

			# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
			split_size = logits.size(0) // 3
			q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

			# performing inner products
			dot_q_d1 = torch.bmm(q_repr.unsqueeze(1), d1_repr.unsqueeze(-1)).squeeze()
			dot_q_d2 = torch.bmm(q_repr.unsqueeze(1), d2_repr.unsqueeze(-1)).squeeze()
			
			# if batch contains only one sample the dotproduct is a scalar rather than a list of tensors
			# so we need to unsqueeze
			if minibatch_samples_number == 1:
				dot_q_d1.unsqueeze(0)
				dot_q_d2.unsqueeze(0)

			# calculating loss
			loss = loss_fn(dot_q_d1, dot_q_d2, targets)

			# calculate l1 loss
			l1_loss = l1_loss_fn(torch.cat([q_repr, d1_repr, d2_repr], 1))
			# calculate balance loss
			balance_loss = balance_loss_fn(logits, device)
			# calculating L0 loss
			l0_q, l0_docs = l0_loss_fn(d1_repr, d2_repr, q_repr)

			# calculating classification accuracy (whether the correct document was classified as more relevant)
			acc = (((dot_q_d1 > dot_q_d2).float() == targets).float() + (
					(dot_q_d2 > dot_q_d1).float() == targets * -1).float()).mean()

			# aggregating losses and running backward pass and update step
			total_loss = loss + l1_loss * l1_scalar + balance_loss * balance_scalar

			total_loss = total_loss * (minibatch_samples_number / batch[1].size(0))

			av_loss.step(loss), av_l1_loss.step(l1_loss), av_balance_loss.step(balance_loss), av_total_loss.step(total_loss), av_l0_q.step(l0_q), av_l0_docs.step(l0_docs), av_acc.step(acc)

			if optim != None:
				total_loss.backward()

		# if we are training, then we perform the backward pass and update step
		if optim != None:
			optim.step()
			optim.zero_grad()

		# get pogress ratio
		samples_trained_ratio = cur_trained_samples / samples_per_epoch

		# check whether we should log (only when in train mode)
		if samples_trained_ratio > current_log_threshold:
			# log
			log_progress(mode, total_trained_samples, cur_trained_samples, samples_per_epoch, av_loss.val, av_l1_loss.val,
						 av_balance_loss.val, av_total_loss.val, av_l0_q.val, av_l0_docs.val, av_acc.val)
			# update log threshold
			current_log_threshold = samples_trained_ratio + log_every_ratio

	# log the values of the final training step
	# log_progress(writer, mode, total_trained_samples, cur_trained_samples, samples_per_epoch, loss, l1_loss,
	# balance_loss, total_loss, l0_q, l0_docs, acc)
	log_progress(mode, total_trained_samples, cur_trained_samples, samples_per_epoch, av_loss.val, av_l1_loss.val,
						 av_balance_loss.val, av_total_loss.val, av_l0_q.val, av_l0_docs.val, av_acc.val, writer=writer)
	return total_trained_samples, av_total_loss.val.item()


def get_all_reprs(model, dataloader, device):
	av_l1_loss, av_l0 = Average(), Average()
	with torch.no_grad():
		model.eval()
		reprs = list()
		ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
			repr_ = model(batch_data_d.to(device), batch_lengths_d.to(device))
			reprs.append(repr_)
			ids += batch_ids_d
			l1, l0 = l1_loss_fn(repr_), l0_loss(repr_)
			av_l1_loss.step(l1), av_l0.step(l0)
		if len(reprs) > 0:
			return reprs, ids, av_l0.val.item(), av_l1_loss.val.item()
		else:
			return None, None, None, None


def get_scores(doc_reprs, doc_ids, q_reprs, max_rank):
	scores = list()
	for batch_q_repr in q_reprs:
		batch_len = len(batch_q_repr)
		# q_score_lists = [ []]*batch_len
		q_score_lists = [[] for i in range(batch_len)]
		for batch_doc_repr in doc_reprs:
			dots_q_d = batch_q_repr @ batch_doc_repr.T
			# appending scores of batch_documents for this batch of queries
			for i in range(batch_len):
				q_score_lists[i] += dots_q_d[i].detach().cpu().tolist()

		# now we will sort the documents by relevance, for each query
		for i in range(batch_len):
			tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(doc_ids, q_score_lists[i])]
			sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)
			if max_rank != -1:
				sorted_by_relevance = sorted_by_relevance[:max_rank]
			scores.append(sorted_by_relevance)
	return scores


def test(model, mode, data_loaders, device, max_rank, total_trained_samples, metric, reset=True, writer=None):
	query_batch_generator, docs_batch_generator = data_loaders[mode]

	if reset:
		docs_batch_generator.reset()
		query_batch_generator.reset()

	scores, q_ids, q_reprs, d_reprs = list(), list(), list(), list()
	av_l1_loss, av_l0_docs, av_l0_query = Average(), Average(), Average()
		

	while True:

		# if return has len == 0 then break
		d_repr, d_ids, l0_docs, l1_loss_docs = get_all_reprs(model, docs_batch_generator, device)
		q_repr, q_ids_q, l0_q, l1_loss_q = get_all_reprs(model, query_batch_generator, device)
		if q_repr is None or d_repr is None:
			break
		
		scores += get_scores(d_repr, d_ids, q_repr, max_rank)
		q_ids += q_ids_q
		av_l0_docs.step(l0_docs)
		av_l0_query.step(l0_q)
		av_l1_loss.step((l1_loss_q + l1_loss_docs)/ 2)
		d_reprs.append(torch.cat(d_repr, 0))
		q_reprs.append(q_repr[0])


	metric_score = metric.score(scores, q_ids)

	if writer != None:
		writer.add_scalar(f'{mode}_l1_loss', av_l1_loss.val , total_trained_samples)

		writer.add_scalar(f'{mode}_L0_query', av_l0_query.val, total_trained_samples)
		writer.add_scalar(f'{mode}_L0_docs', av_l0_docs.val, total_trained_samples)

		writer.add_scalar(f'{metric.name}', metric_score, total_trained_samples)
	print(f'{mode} -  {metric.name}: {metric_score}')
	return scores, q_reprs, d_reprs, q_ids, d_ids, metric_score


def run(model, dataloaders, optim, loss_fn, epochs, writer, device, model_folder,
		  l1_scalar=1, balance_scalar=1, patience=2, samples_per_epoch_train=10000, samples_per_epoch_val=20000,
		  bottleneck_run=False, log_every_ratio=0.01, max_rank=1000, metric=None,
		  sparse_dimensions=1000, validate=True, max_samples_per_gpu = 16, n_gpu = 1):
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
	eval_mode='min' if validate else 'max'
	early_stopper = EarlyStopping(patience=patience, mode=eval_mode)
	total_trained_samples = 0

	# initialize data loader for the first epoch
	if total_trained_samples == 0:
		batch_iterator_train = iter(dataloaders['train'])
		if validate:
			batch_iterator_val = iter(dataloaders['val'])

	for epoch in range(1, epochs + 1):

		if early_stopper.stop:
			print(f"Early Stopping at Epoch: {epoch-1}!")
			break

		print('Epoch', epoch)
		# training
		with torch.enable_grad():
			model.train()
			total_trained_samples, _ = run_epoch(model, 'train', dataloaders, batch_iterator_train, loss_fn, epoch,
												 writer,
												 l1_scalar, balance_scalar, total_trained_samples, device,
												 optim=optim, samples_per_epoch=samples_per_epoch_train,
												 log_every_ratio=log_every_ratio, max_samples_per_gpu = max_samples_per_gpu, n_gpu = n_gpu)

		# evaluation
		with torch.no_grad():
			model.eval()

			if bottleneck_run:
				break
			else:
				
				if validate:
					_, val_total_loss = run_epoch(model, 'val', dataloaders, batch_iterator_val, loss_fn, epoch, writer, l1_scalar, balance_scalar, total_trained_samples, device,
						optim=None, samples_per_epoch=samples_per_epoch_val, log_every_ratio=log_every_ratio, max_samples_per_gpu = max_samples_per_gpu, n_gpu = n_gpu)


				# Run also proper evaluation script
				_, q_repr, d_repr, q_ids, _, metric_score = test(model, 'test', dataloaders, device, max_rank,
																	total_trained_samples, metric, writer=writer)
	
				# plot stats
				plot_ordered_posting_lists_lengths(model_folder, q_repr, 'query')
				plot_histogram_of_latent_terms(model_folder, q_repr, sparse_dimensions, 'query')
				plot_ordered_posting_lists_lengths(model_folder, d_repr, 'docs')
				plot_histogram_of_latent_terms(model_folder, d_repr, sparse_dimensions, 'docs')

				if validate:
					metric_score = val_total_loss
				else:
					metric_score = metric_score
				

				# check for early stopping
				if not early_stopper.step(metric_score) :
					print(f'Best model at current epoch {epoch}, av value: {metric_score}')
					# save best model so far to file
					torch.save(model, f'{model_folder}/best_model.model')
				


	if not bottleneck_run:
		# load best model
		model = torch.load(f'{model_folder}/best_model.model')

	return model, early_stopper.best, total_trained_samples
