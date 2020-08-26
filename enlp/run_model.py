import os.path
import subprocess

import numpy as np
import torch

from enlp.utils import l1_loss_fn, l0_loss_fn, balance_loss_fn, l0_loss, plot_histogram_of_latent_terms, \
	plot_ordered_posting_lists_lengths, Average, EarlyStopping, split_batch_to_minibatches, split_batch_to_minibatches_bert_interaction


def log_progress(mode, total_trained_samples, currently_trained_samples, samples_per_epoch, loss, l1_loss,
				 balance_loss, total_loss, l0_q, l0_docs, acc, writer=None):
	print("{}  {}/{} total loss: {:.4f}, task loss: {:.4f}, l1 loss: {:.4f}, balance loss: {:.4f}, acc: {:.4f}".format(mode, currently_trained_samples, samples_per_epoch, total_loss, loss, l1_loss, balance_loss, acc))
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
			continue

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

		if isinstance(model, torch.nn.DataParallel):
			model_type = model.module.model_type
		else:
			model_type = model.model_type


		if model_type == "bert-interaction" or model_type == "bert-interaction_pair_wise":
			_, _, _, batch_targets = batch
		else:
			_, batch_targets, _ = batch


		if optim != None:
			optim.zero_grad()

		if model_type == "bert-interaction":
			minibatches = split_batch_to_minibatches_bert_interaction(batch, max_samples_per_gpu = max_samples_per_gpu, n_gpu=n_gpu, pairwise_training = False)
		elif model_type == "bert-interaction_pair_wise":
			minibatches = split_batch_to_minibatches_bert_interaction(batch, max_samples_per_gpu = max_samples_per_gpu, n_gpu=n_gpu, pairwise_training = True)
		else:
			minibatches = split_batch_to_minibatches(batch, max_samples_per_gpu = max_samples_per_gpu, n_gpu=n_gpu)

		batch_samples_number = 0


		for minibatch in minibatches:

			torch.cuda.empty_cache()
			# move to device
			minibatch = [item.to(device) for item in minibatch]

			if model_type == "bert-interaction" or model_type == "bert-interaction_pair_wise":
				input_ids, attention_masks, token_type_ids, targets = minibatch
			else:
				data, targets, lengths = minibatch

			# get number of samples within the minibatch
			minibatch_samples_number = targets.size(0)

			batch_samples_number += minibatch_samples_number

			# update the number of trained samples in this epoch
			cur_trained_samples += minibatch_samples_number
			# update the total number of trained samples
			total_trained_samples += minibatch_samples_number


			# if the model provides an indipendednt representation for the input (query/doc)
			if model_type == "representation-based":

				# forward pass (inputs are concatenated in the form [q1, q2, ..., q1d1, q2d1, ..., q1d2, q2d2, ...])
				logits = model(data, lengths)
				# moving targets also to the appropriate device
				# targets = targets.to(device)

				# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
				split_size = logits.size(0) // 3
				q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

				# performing inner products
				score_q_d1 = torch.bmm(q_repr.unsqueeze(1), d1_repr.unsqueeze(-1)).squeeze()
				score_q_d2 = torch.bmm(q_repr.unsqueeze(1), d2_repr.unsqueeze(-1)).squeeze()

				# if batch contains only one sample the dotproduct is a scalar rather than a list of tensors
				# so we need to unsqueeze
				if minibatch_samples_number == 1:
					score_q_d1 = score_q_d1.unsqueeze(0)
					score_q_d2 = score_q_d2.unsqueeze(0)

				# calculate l1 loss
				l1_loss = l1_loss_fn(torch.cat([q_repr, d1_repr, d2_repr], 1))
				# calculate balance loss
				balance_loss = balance_loss_fn(logits, device)
				# calculating L0 loss
				l0_q, l0_docs = l0_loss_fn(d1_repr, d2_repr, q_repr)

			# if the model provides a score for a document and a query
			elif model_type == "interaction-based":
				split_size = data.size(0) // 3
				q_repr, doc1, doc2 = torch.split(data, split_size)
				lengths_q, lengths_d1, lengths_d2 = torch.split(lengths, split_size)
				#score_q_d1 = model(q_repr, doc1, lengths_q, lengths_d1)
				#score_q_d2 = model(q_repr, doc2, lengths_q, lengths_d2)
				d_concat = torch.cat((doc1, doc2), 0)
				q_concat = torch.cat((q_repr, q_repr), 0)
				lengths_q_concat = torch.cat((lengths_q, lengths_q), 0)
				lengths_d_concat = torch.cat((lengths_d1, lengths_d2), 0)
				scores = model(q_concat, d_concat, lengths_q_concat, lengths_d_concat)
				split_size = scores.size(0) // 2
				score_q_d1, score_q_d2 = torch.split(scores, split_size)
				#print(score_q_d2.shape)
				#print(targets.shape)
				#print(targets)

			elif model_type == "bert-interaction":
				# apply model
				relevance_out = model(input_ids, attention_masks, token_type_ids)

			elif model_type == "bert-interaction_pair_wise":
				# every second sample corresponds to d2, so we split inputs accordingly

				num_of_samples = int(input_ids.size(0) / 2)

				# every second sample corresponds to d2, so we split inputs accordingly
				input_ids = input_ids.view(num_of_samples,2, -1)
				attention_masks = attention_masks.view(num_of_samples,2,-1)
				token_type_ids = token_type_ids.view(num_of_samples,2,-1)

				qd1_input_ids = input_ids[:,0]
				qd2_input_ids = input_ids[:,1]

				qd1_attention_masks = attention_masks[:,0]
				qd2_attention_masks = attention_masks[:,1]

				qd1_token_type_ids = token_type_ids[:,0]
				qd2_token_type_ids = token_type_ids[:,1]

				score_q_d1 = model(qd1_input_ids, qd1_attention_masks, qd1_token_type_ids)
				score_q_d2 = model(qd2_input_ids, qd2_attention_masks, qd2_token_type_ids)

				score_q_d1 = torch.tanh(score_q_d1)
				score_q_d2 = torch.tanh(score_q_d2)

			else:
				raise ValueError(f"run_model.py , model_type not properly defined!: {model_type}")

			# the following metrics are not calculated in case we do not run a "representation-based" model
			if model_type != "representation-based":
				# calculate l1 loss
				l1_loss = torch.tensor(0)
				# calculate balance loss
				balance_loss = torch.tensor(0)
				# calculating L0 loss
				l0_q, l0_docs = torch.tensor(0), torch.tensor(0)

			if model_type == "bert-interaction":
				loss = loss_fn(relevance_out, targets)
				acc = ((relevance_out[:, 1] > relevance_out[:, 0]).int() == targets).float().mean()
			else:
				# calculating loss
				loss = loss_fn(score_q_d1, score_q_d2, targets)
				# calculating classification accuracy (whether the correct document was classified as more relevant)
				acc = (((score_q_d1 > score_q_d2).float() == targets).float() + (
						(score_q_d2 >= score_q_d1).float() == targets * -1).float()).mean()

			# aggregating losses and running backward pass and update step
			total_loss = loss + l1_loss * l1_scalar + balance_loss * balance_scalar

			total_loss = total_loss * (minibatch_samples_number / batch_targets.size(0))

			av_loss.step(loss), av_l1_loss.step(l1_loss), av_balance_loss.step(balance_loss), av_total_loss.step(total_loss), av_l0_q.step(l0_q), av_l0_docs.step(l0_docs), av_acc.step(acc)

			if optim != None:
				total_loss.backward()

			if next(model.parameters()).is_cuda:
				torch.cuda.empty_cache()

		# if we are training, then we perform the backward pass and update step
		if optim != None:
			optim.step()
			optim.zero_grad()

		if next(model.parameters()).is_cuda:
			torch.cuda.empty_cache()

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

	return total_trained_samples, av_total_loss.val.item(), av_loss.val.item(), av_l1_loss.val.item(), av_l0_q.val.item(), av_l0_docs.val.item(), av_acc.val.item()



def get_dot_scores(doc_reprs, doc_ids, q_reprs, max_rank):
	scores = list()
	for batch_q_repr in q_reprs:
		batch_len = len(batch_q_repr)
		# q_score_lists = [ []]*batch_len
		q_score_lists = [[] for i in range(batch_len)]
		for batch_doc_repr in doc_reprs:
			dots_q_d = batch_q_repr @ batch_doc_repr.T
			# appending scores of batch_documents for this batch of queries
			for i in range(batch_len):
				q_score_lists[i] += list(dots_q_d[i])

		# now we will sort the documents by relevance, for each query
		for i in range(batch_len):
			tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(doc_ids, q_score_lists[i])]
			sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)
			if max_rank != -1:
				sorted_by_relevance = sorted_by_relevance[:max_rank]
			scores.append(sorted_by_relevance)
	return scores

def get_rerank_representations(model, dataloader, device):
	av_l1_loss_q, av_l0_q, av_l1_loss_d, av_l0_d = Average(), Average(), Average(), Average()
	reprs_q, ids_q, reprs_d, ids_d = list(), list(), list(), list()

	for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
		data_q, length_q, batch_data_d, batch_lengths_d  = data_q.to(device), length_q.to(device), batch_data_d.to(device), batch_lengths_d.to(device)
		repr_d = model(batch_data_d, batch_lengths_d)
		reprs_d.append(repr_d.detach().cpu().numpy())
		ids_d += batch_ids_d
		l1_d, l0_d = l1_loss_fn(repr_d), l0_loss(repr_d)
		av_l1_loss_d.step(l1_d), av_l0_d.step(l0_d)

		# we want to return each query only once for all ranked documents for this query
		if len(reprs_q) == 0:
			repr_q = model(data_q, length_q)
			reprs_q.append(repr_q.detach().cpu().numpy())
			ids_q += q_id
			l1_q, l0_q = l1_loss_fn(repr_q), l0_loss(repr_q)
			av_l1_loss_q.step(l1_q), av_l0_q.step(l0_q)
	if len(reprs_d) > 0:
		return reprs_q, ids_q, av_l0_q.val.item(), av_l1_loss_q.val.item(), reprs_d, ids_d, av_l0_d.val.item(), av_l1_loss_d.val.item()
	else:
		return None, None, None, None, None, None, None, None


def scores_representation_based(model, dataloader, device, writer, max_rank, total_trained_samples, reset, model_folder, mode, plot=True):

	scores, q_ids, q_reprs, d_reprs = list(), list(), list(), list()
	av_l1_loss, av_l0_docs, av_l0_query = Average(), Average(), Average()

	if reset:
		dataloader.reset()

	while True:
		# if return has len == 0 then break
		q_repr, q_id, l0_q, l1_loss_q, d_repr, d_ids, l0_docs, l1_loss_docs = get_rerank_representations(model, dataloader, device)
		if q_repr is None or d_repr is None:
			break
		scores += get_dot_scores(d_repr, d_ids, q_repr, max_rank)
		q_ids += q_id
		av_l0_docs.step(l0_docs)
		av_l0_query.step(l0_q)
		av_l1_loss.step((l1_loss_q + l1_loss_docs)/ 2)
		d_reprs.append(np.concatenate(d_repr, 0))
		q_reprs.append(q_repr[0])

	if plot:
		# plot stats
		plot_ordered_posting_lists_lengths(model_folder, q_reprs, 'query')
		plot_histogram_of_latent_terms(model_folder, q_reprs, 'query')
		plot_ordered_posting_lists_lengths(model_folder, d_reprs, 'docs')
		plot_histogram_of_latent_terms(model_folder, d_reprs, 'docs')

	if writer != None:
		writer.add_scalar(f'{mode}_l1_loss', av_l1_loss.val , total_trained_samples)
		writer.add_scalar(f'{mode}_L0_query', av_l0_query.val, total_trained_samples)
		writer.add_scalar(f'{mode}_L0_docs', av_l0_docs.val, total_trained_samples)
	return scores, q_ids


def get_repr_inter(model, dataloader, device, max_rank):
	scores, q_ids, d_ids  = list(), list(), list()
	for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
			data_q, length_q, batch_data_d, batch_lengths_d  = data_q.to(device), length_q.to(device), batch_data_d.to(device), batch_lengths_d.to(device)
			# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
			# repeat query for each document
			n_repeat = batch_data_d.shape[0]
			batch_data_q = data_q.repeat(n_repeat,1).to(device)
			lengths_q = length_q.repeat(n_repeat).to(device)
			score = model(batch_data_q, batch_data_d, lengths_q, batch_lengths_d)
			scores += score.detach().cpu().tolist()
			d_ids += batch_ids_d
			# we want to return each query only once for all ranked documents for this query
			if len(q_ids) == 0:
				q_ids += q_id
	if len(q_ids) < 1:
		return None, None, None
	scores = np.array(scores).flatten()
	tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(d_ids, scores)]
	sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)
	if max_rank != -1:
		sorted_by_relevance = sorted_by_relevance[:max_rank]
	return q_ids, d_ids, sorted_by_relevance

def scores_interaction_based(model, dataloader, device, reset, max_rank):
	scores, q_ids = list(), list()
	if reset:
		dataloader.reset()
	while True:
		q_id, d_ids, score = get_repr_inter(model, dataloader, device, max_rank)
		if score is None:
			break
		scores.append(score)
		q_ids += q_id
	return scores, q_ids

def scores_bert_interaction(model, dataloader, device, reset, max_rank, pairwise = False):
	all_scores, all_q_ids = [], []
	if reset:
		dataloader.reset()

	while True:

		scores, q_ids, d_ids = [], [], []
		# for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator_bert_interaction():
		for q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids in dataloader.batch_generator_bert_interaction():
			# move data to device
			batch_input_ids, batch_attention_masks, batch_token_type_ids = batch_input_ids.to(device), batch_attention_masks.to(device), batch_token_type_ids.to(device)
			# propagate data through model
			model_out = model(batch_input_ids, batch_attention_masks, batch_token_type_ids)
			if pairwise == False:
				# After retrieving model's output, we apply softax and keep the second dimension that represents the relevance probability
				score = torch.softmax(model_out, dim=-1)[:,1]
			else:
				score = torch.tanh(model_out)

			scores += score.detach().cpu().tolist()
			d_ids += d_batch_ids
			# we want to return each query only once for all ranked documents for this query
			if len(q_ids) == 0:
				q_ids += q_id

		if len(q_ids) < 1:
			return all_scores, all_q_ids
		scores = np.array(scores).flatten()
		tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(d_ids, scores)]
		sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)

		if max_rank != -1:
			sorted_by_relevance = sorted_by_relevance[:max_rank]

		if sorted_by_relevance is None:
			break

		all_scores.append(sorted_by_relevance)
		all_q_ids += q_id
	return all_scores, all_q_ids



# returns metric score if metric != None
# if metric = None returns scores, qiids

def test(model, mode, data_loaders, device, max_rank, total_trained_samples, model_folder, reset=True, writer=None, metric=None):
	if isinstance(model, torch.nn.DataParallel):
		model_type = model.module.model_type
	else:
		model_type = model.model_type
	# if the model provides an indipendednt representation for the input (query/doc)
	if model_type == "representation-based":
		scores, q_ids = scores_representation_based(model, data_loaders[mode], device, writer, max_rank, total_trained_samples, reset, model_folder, mode, plot=True)
	elif model_type == "interaction-based":
		scores, q_ids = scores_interaction_based(model, data_loaders[mode], device, reset, max_rank)
	elif model_type == "bert-interaction":
		scores, q_ids = scores_bert_interaction(model, data_loaders[mode], device, reset, max_rank, pairwise = False)
	elif model_type == "bert-interaction_pair_wise":
		scores, q_ids = scores_bert_interaction(model, data_loaders[mode], device, reset, max_rank, pairwise = True)
	else:
		raise ValueError(f"run_model.py , model_type not properly defined!: {model_type}")
	if metric:
		metric_score = metric.score(scores, q_ids)
		if writer:
			writer.add_scalar(f'{metric.name}', metric_score, total_trained_samples)
		print(f'{mode} -  {metric.name}: {metric_score}')
		return metric_score
	else:
		return scores, q_ids

def run(model, dataloaders, optim, loss_fn, epochs, writer, device, model_folder,
		  l1_scalar=1, balance_scalar=1, patience=2, samples_per_epoch_train=10000, samples_per_epoch_val=20000,
		  bottleneck_run=False, log_every_ratio=0.01, max_rank=1000, metric=None,
		  sparse_dimensions=1000, validate=True, max_samples_per_gpu = 16, n_gpu = 1, telegram=False):
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
			total_trained_samples, train_total_loss, train_task_loss, train_l1_loss, train_l0_q, train_l0_docs, train_acc = run_epoch(model, 'train',
												 dataloaders, batch_iterator_train, loss_fn, epoch, writer,
												 l1_scalar, balance_scalar, total_trained_samples, device,
												 optim=optim, samples_per_epoch=samples_per_epoch_train,
												 log_every_ratio=log_every_ratio, max_samples_per_gpu = max_samples_per_gpu, n_gpu = n_gpu)

			telegram_message = f'Train:\nTotal loss {round(train_total_loss, 4)}\nTrain task_loss {round(train_task_loss, 4)}\nl1_loss {round(train_l1_loss, 4)}\nL0_query {round(train_l0_q, 4)}\nL0_docs {round(train_l0_docs, 4)}\nacc {round(train_acc, 4)}'
			telegram_message = model_folder + '\n' + telegram_message
			if telegram:
				subprocess.run(["bash", "telegram.sh", "-c", "-462467791", telegram_message])

			# in case the model has gone completely wrong, stop training
			if train_acc < 0.3:
				print('Ending training train because train accurracy is < 0.3!')
				#break

		# evaluation
		with torch.no_grad():
			model.eval()
			if bottleneck_run:
				print('Bottleneck run, stopping now!')
				break

			else:

				if validate:
					_, val_total_loss, val_task_loss, val_l1_loss, val_l0_q, val_l0_docs, val_acc = run_epoch(model, 'val', dataloaders, batch_iterator_val, loss_fn, epoch, writer, l1_scalar, balance_scalar, total_trained_samples, device,
						optim=None, samples_per_epoch=samples_per_epoch_val, log_every_ratio=log_every_ratio, max_samples_per_gpu = max_samples_per_gpu, n_gpu = n_gpu)

					if telegram:
						telegram_message = f'Validation:\nTotal loss {round(val_total_loss, 4)}\nTrain task_loss {round(val_task_loss, 4)}\nl1_loss {round(val_l1_loss, 4)}\nL0_query {round(val_l0_q, 4)}\nL0_docs {round(val_l0_docs, 4)}\nacc {round(val_acc, 4)}'
						telegram_message = model_folder + '\n' + telegram_message
						subprocess.run(["bash", "telegram.sh", "-c", "-462467791", telegram_message])

				# Run also proper evaluation script
				print('Running test: ')
				metric_score = test(model, 'test', dataloaders, device, max_rank,
																	total_trained_samples, model_folder=model_folder, writer=writer, metric=metric)

				if telegram:
					telegram_message = model_folder + '\n' + f'Test Metric Score:\n{metric_score}'
					subprocess.run(["bash", "telegram.sh", "-c", "-462467791", telegram_message])

				if validate:
					metric_score = val_total_loss
				else:
					metric_score = metric_score


				# check for early stopping
				if not early_stopper.step(metric_score) :
					print(f'Best model at current epoch {epoch}, av value: {metric_score}')
					# save best model so far to file
					torch.save(model.state_dict(), f'{model_folder}/best_model.model')

	return early_stopper.best, total_trained_samples
