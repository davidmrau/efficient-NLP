import os.path
import subprocess

import numpy as np
import torch
import copy
import time

from enlp.utils import l1_loss_fn, l0_loss_fn, balance_loss_fn, l0_loss, plot_histogram_of_latent_terms, \
	plot_ordered_posting_lists_lengths, Average, EarlyStopping


def log_progress(mode, total_trained_samples, currently_trained_samples, samples_per_epoch, loss, l1_loss,
				 balance_loss, total_loss, l0_q, l0_docs, acc, writer=None):
	print("{}  {}/{} total loss: {:.4f}, task loss: {:.4f}, l1 loss: {:.4f}, balance loss: {:.4f}, acc: {:.4f}".format(
		mode, currently_trained_samples, samples_per_epoch, total_loss, loss, l1_loss, balance_loss, acc))
	if writer:
		# update tensorboard
		writer.add_scalar(f'{mode}/task_loss', loss, total_trained_samples)
		writer.add_scalar(f'{mode}/l1_loss', l1_loss, total_trained_samples)
		writer.add_scalar(f'{mode}/balance_loss', balance_loss, total_trained_samples)
		writer.add_scalar(f'{mode}/total_loss', total_loss, total_trained_samples)
		writer.add_scalar(f'{mode}/L0_query', l0_q, total_trained_samples)
		writer.add_scalar(f'{mode}/L0_docs', l0_docs, total_trained_samples)
		writer.add_scalar(f'{mode}/acc', acc, total_trained_samples)


def run_epoch(model, mode, dataloader, batch_iterator, loss_fn, writer, l1_scalar, balance_scalar,
			  total_trained_samples, device, optim=None, samples_per_epoch=10000, log_every_ratio=0.01, sub_batch_size=None):
	"""Train 1 epoch, and evaluate.

	Returns
	-------
	av_loss                 : (float) average loss
	total_trained_samples    : (int) the total number of training steps performed over all epochs
	type
		Description of returned object.
	"""
	log_every_ratio = max(dataloader[mode].batch_size / samples_per_epoch, log_every_ratio)

	current_log_threshold = log_every_ratio

	cur_trained_samples = 0

	av_loss, av_l1_loss, av_balance_loss, av_total_loss, av_l0_q, av_l0_docs, av_acc = Average(), Average(), Average(), Average(), Average(), Average(), Average()
	accumulation_steps = 0
	while cur_trained_samples < samples_per_epoch:

		try:
			batch = next(batch_iterator)
		except StopIteration:
			# StopIteration is thrown if dataset ends
			# reinitialize data loader
			batch_iterator = iter(dataloader[mode])
			continue

		# For the weak supervision setting, some doc/queries are empty, rsulting to that sample being None
		# If all samples are None within a batch, then the batch is None
		if batch is None:
			continue

		if isinstance(model, torch.nn.DataParallel):
			model_type = model.module.model_type
		else:
			model_type = model.model_type

		batch = [item.to(device) for item in batch]
		if model_type == "bert-interaction":
			input_ids, attention_masks, token_type_ids, targets = batch
		else:
			q, doc1, doc2, lengths_q, lengths_d1, lengths_d2, targets = batch



		# get number of samples within the minibatch
		batch_samples_number = targets.size(0)
		# update the number of trained samples in this epoch
		cur_trained_samples += batch_samples_number
		# update the total number of trained samples
		total_trained_samples += batch_samples_number


		l1_loss = torch.tensor(0)
		# calculate balance loss
		balance_loss = torch.tensor(0)
		# calculating L0 loss
		l0_q, l0_docs = torch.tensor(0), torch.tensor(0)

		# if the model provides an independent representation for the input (query/doc)
		if model_type == "representation-based":
			q_repr = model(q, lengths_q)
			d1_repr = model(doc1, lengths_d1)
			d2_repr = model(doc2, lengths_d2)


			# performing inner products
			score_q_d1 = torch.bmm(q_repr.unsqueeze(1), d1_repr.unsqueeze(-1)).squeeze()
			score_q_d2 = torch.bmm(q_repr.unsqueeze(1), d2_repr.unsqueeze(-1)).squeeze()

			# if batch contains only one sample the dotproduct is a scalar rather than a list of tensors
			# so we need to unsqueeze
			if batch_samples_number == 1:
				score_q_d1 = score_q_d1.unsqueeze(0)
				score_q_d2 = score_q_d2.unsqueeze(0)

			# calculate l1 loss
			logits_comb = torch.cat([q_repr, d1_repr, d2_repr], 0)
			l1_loss = l1_loss_fn(logits_comb)
			# calculate balance loss
			balance_loss = balance_loss_fn(logits_comb, device)
			# calculating L0 loss
			l0_q, l0_docs = l0_loss_fn(d1_repr, d2_repr, q_repr)

			# calculating loss
			loss = loss_fn(score_q_d1, score_q_d2, targets)
			# calculating classification accuracy (whether the correct document was classified as more relevant)
			targets[ targets == -1 ] = 0
			acc = (((score_q_d1 > score_q_d2).float() == targets).float()).mean()

		# if the model provides a score for a document and a query
		elif model_type == "interaction-based":
			score_q_d1 = model(q, doc1, lengths_q, lengths_d1)
			score_q_d2 = model(q, doc2, lengths_q, lengths_d2)
			# calculate l1 loss


			# calculating loss
			loss = loss_fn(score_q_d1, score_q_d2, targets)
			# calculating classification accuracy (whether the correct document was classified as more relevant)
			targets[targets == -1] = 0
			acc = (((score_q_d1 > score_q_d2).float() == targets).float()).mean()
		elif model_type == "rank-interaction":
			score_q_d1 = model(q, doc1, lengths_q, lengths_d1)
			score_q_d2 = model(q, doc2, lengths_q, lengths_d2)
			# calculate l1 los
			targets[targets == -1] = 0
			targets_2 = (~targets.bool()).int().float()
			relevance_comb = torch.cat((score_q_d1, score_q_d2), 0)
			targets_comb = torch.cat((targets, targets_2), 0).long()
			# calculating loss
			loss = loss_fn(relevance_comb, targets_comb)
			# calculating classification accuracy (whether the correct document was classified as more relevant)

			acc = ((relevance_comb[:, 1] > relevance_comb[:, 0]).int() == targets_comb).float().mean()

		elif model_type == 'rank_prob':
			score_q_d = model(q, doc1, doc2, lengths_q, lengths_d1, lengths_d2)
			# calculate l1 los
			targets[targets == -1] = 0
			loss = loss_fn(score_q_d, targets.long())
			acc = ((score_q_d[:, 1] > score_q_d[:, 0]).int() == targets).float().mean()

		elif model_type == "bert-interaction":
			# apply model
			relevance_out = model(input_ids, attention_masks, token_type_ids)

			loss = loss_fn(relevance_out, targets)
			acc = ((relevance_out[:, 1] > relevance_out[:, 0]).int() == targets).float().mean()

		else:
			raise ValueError(f"run_model.py , model_type not properly defined!: {model_type}")

		# aggregating losses and running backward pass and update step
		total_loss = loss + l1_loss * l1_scalar + balance_loss * balance_scalar
		#total_loss = total_loss  # / accumulation_steps

		av_loss.step(loss), av_l1_loss.step(l1_loss), av_balance_loss.step(balance_loss), av_total_loss.step(
			total_loss), av_l0_q.step(l0_q), av_l0_docs.step(l0_docs), av_acc.step(acc)
		if sub_batch_size:
			total_loss = total_loss / (dataloader[mode].batch_size / sub_batch_size)
			accumulation_steps += sub_batch_size
		if optim != None:
			total_loss.backward()
		# if we are training, then we perform the backward pass and update step
		if optim != None:
			if accumulation_steps % dataloader[mode].batch_size  == 0 or sub_batch_size == None:
				optim.step()
				optim.zero_grad()
		# get pogress ratio
		samples_trained_ratio = cur_trained_samples / samples_per_epoch

		# check whether we should log (only when in train mode)
		if samples_trained_ratio > current_log_threshold:
			# log
			log_progress(mode, total_trained_samples, cur_trained_samples, samples_per_epoch, av_loss.val,
						 av_l1_loss.val,
						 av_balance_loss.val, av_total_loss.val, av_l0_q.val, av_l0_docs.val, av_acc.val, writer=writer)
			# update log threshold
			current_log_threshold = samples_trained_ratio + log_every_ratio

	# log the values of the final training step
	log_progress(mode, total_trained_samples, cur_trained_samples, samples_per_epoch, av_loss.val, av_l1_loss.val,
				 av_balance_loss.val, av_total_loss.val, av_l0_q.val, av_l0_docs.val, av_acc.val, writer=writer)

	return batch_iterator, total_trained_samples, av_total_loss.val, av_loss.val, av_l1_loss.val, av_l0_q.val, av_l0_docs.val, av_acc.val


def get_dot_scores(doc_reprs, doc_ids, q_reprs):
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
			tuples_of_doc_ids_and_scores = list(zip(doc_ids, q_score_lists[i]))

			sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)
			scores.append(sorted_by_relevance)
	return scores


def get_rerank_reprs(model, dataloader, device):
	av_l1_loss_q, av_l0_q, av_l1_loss_d, av_l0_d = Average(), Average(), Average(), Average()
	reprs_q, ids_q, reprs_d, ids_d = list(), list(), list(), list()

	for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
		#data_q, length_q, batch_data_d, batch_lengths_d = data_q.to(device), length_q.to(device), batch_data_d.to(device), batch_lengths_d.to(device)
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


def scores_representation_based(model, dataloader, device, writer, total_trained_samples, reset, model_folder,
								mode, plot=True):
	scores, q_ids, q_reprs, d_reprs = list(), list(), list(), list()
	av_l1_loss, av_l0_docs, av_l0_query = Average(), Average(), Average()

	if reset:
		dataloader.reset()

	while True:
		# if return has len == 0 then break
		q_repr, q_id, l0_q, l1_loss_q, d_repr, d_ids, l0_docs, l1_loss_docs = get_rerank_reprs(model, dataloader, device)
		if q_repr is None or d_repr is None:
			break
		scores += get_dot_scores(d_repr, d_ids, q_repr)
		q_ids += q_id
		av_l0_docs.step(l0_docs)
		av_l0_query.step(l0_q)
		av_l1_loss.step((l1_loss_q + l1_loss_docs) / 2)
		d_reprs.append(np.concatenate(d_repr, 0))
		q_reprs.append(q_repr[0])

	if plot:
		# plot stats
		plot_ordered_posting_lists_lengths(model_folder, q_reprs, 'query')
		plot_histogram_of_latent_terms(model_folder, q_reprs, 'query')
		plot_ordered_posting_lists_lengths(model_folder, d_reprs, 'docs')
		plot_histogram_of_latent_terms(model_folder, d_reprs, 'docs')

	if writer != None:
		writer.add_scalar(f'{mode}/l1_loss', av_l1_loss.val, total_trained_samples)
		writer.add_scalar(f'{mode}/L0_query', av_l0_query.val, total_trained_samples)
		writer.add_scalar(f'{mode}/L0_docs', av_l0_docs.val, total_trained_samples)
	return scores, q_ids


def get_score_inter(model, dataloader, device, classifier=False):
	scores, q_ids, d_ids = list(), list(), list()
	for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
		##data_q, length_q, batch_data_d, batch_lengths_d = data_q.to(device), length_q.to(device), batch_data_d.to(device), batch_lengths_d.to(device)

		# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
		# repeat query for each document
		n_repeat = batch_data_d.shape[0]
		batch_data_q = data_q.repeat(n_repeat, 1)
		lengths_q = length_q.repeat(n_repeat)
		score = model(batch_data_q, batch_data_d, lengths_q, batch_lengths_d)
		if classifier:
			score = torch.softmax(score, dim=-1)[:, 1]
		scores += score.detach().cpu().tolist()
		d_ids += batch_ids_d
		# we want to return each query only once for all ranked documents for this query
		if len(q_ids) == 0:
			q_ids += q_id
	if len(q_ids) < 1:
		return None, None, None
	scores = np.array(scores).flatten()
	tuples_of_doc_ids_and_scores = list(zip(d_ids, scores))
	sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)
	return q_ids, d_ids, sorted_by_relevance


def get_accumulated_data(dataloader):
	accumulated_data = list()
	for d in dataloader.batch_generator():
		accumulated_data.append(d)
	if len(accumulated_data) < 1:
		return None
	return accumulated_data


def get_scores_all_comb(model, accumulated_data, device):
	with torch.no_grad():
		scores, q_ids, d_ids = list(), list(), list()
		for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in accumulated_data:
			#data_q, length_q, batch_data_d, batch_lengths_d = data_q.to(device), length_q.to(device), batch_data_d.to(device), batch_lengths_d.to(device)
			print(data_q.device, batch_data_d.device) 
			#q_av = model.module.get_av_repr(data_q, length_q)
			t0 = time.time()
			for data_d, length_d in zip(batch_data_d, batch_lengths_d):
				#d_av = model.module.get_av_repr(data_d.unsqueeze(0), length_d.unsqueeze(0))
				score_doc = list()
				for _, _, _, _, batch_data_d_2, batch_lengths_d_2 in accumulated_data:
					#batch_data_d_2, batch_lengths_d_2 = batch_data_d_2.to(device), batch_lengths_d_2.to(device)
					#d_av_2 = model(q=batch_data_d_2, doc1=batch_lengths_d_2, get_av_repr=True)
					#relevance = model(q=q_av, doc1=d_av, doc2=d_av_2, av_provided=True)
					n_repeat = batch_data_d_2.shape[0]
					batch_data_q = data_q.repeat(n_repeat, 1)
					batch_data_d = data_d.repeat(n_repeat, 1)
					lengths_q = length_q.repeat(n_repeat)
					lengths_d = length_d.repeat(n_repeat)
					#relevance = model(q=q_av, doc1=d_av, doc2=d_av_2, av_provided=True)
					relevance = model(q=batch_data_q, doc1=batch_data_d, doc2=batch_data_d_2, lengths_q=lengths_q, lengths_d1=lengths_d, lengths_d2=batch_lengths_d_2)
					relevance = torch.softmax(relevance, dim=-1)[:, 1]
					relevance = relevance.detach().cpu().tolist()
					score_doc += relevance
				scores.append(np.mean(score_doc))
			t1 = time.time()
			print(t1-t0)
			d_ids += batch_ids_d
		# we want to return each query only once for all ranked documents for this query
		if len(q_ids) == 0:
			q_ids += q_id

		# now we will sort the documents by relevance, for each query
		scores = np.array(scores).flatten()
		tuples_of_doc_ids_and_scores = list(zip(d_ids, scores))
		sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)
	return q_ids, d_ids, sorted_by_relevance



def scores_all_comb(model, dataloader, device, reset, metric=None):
	scores, q_ids = list(), list()
	if reset:
		dataloader.reset()
	while True:
		accumulated_data = get_accumulated_data(dataloader)
		if accumulated_data is None:
			break
		q_id, d_ids, score = get_scores_all_comb(model, accumulated_data, device)
		if metric:
			metric_score = metric.score([score], q_id, save_path=metric.ranking_file_path+ str(q_id[-1]))
		scores.append(score)
		q_ids += q_id

	return scores, q_ids


def scores_interaction_based(model, dataloader, device, reset, classifier=False):
	scores, q_ids = list(), list()
	if reset:
		dataloader.reset()
	while True:
		q_id, d_ids, score = get_score_inter(model, dataloader, device, classifier=classifier)
		if score is None:
			break
		scores.append(score)
		q_ids += q_id
	return scores, q_ids


def scores_bert_interaction(model, dataloader, device, reset):
	all_scores, all_q_ids = [], []
	if reset:
		dataloader.reset()

	while True:

		scores, q_ids, d_ids = [], [], []
		# for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator_bert_interaction():
		for q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids in dataloader.batch_generator_bert_interaction():
			# move data to device
			#batch_input_ids, batch_attention_masks, batch_token_type_ids = batch_input_ids.to(device), batch_attention_masks.to(device), batch_token_type_ids.to(device)
			# propagate data through model
			model_out = model(batch_input_ids, batch_attention_masks, batch_token_type_ids)
			# After retrieving model's output, we apply softax and keep the second dimension that represents the relevance probability
			score = torch.softmax(model_out, dim=-1)[:, 1]
			scores += score.detach().cpu().tolist()
			d_ids += d_batch_ids
			# we want to return each query only once for all ranked documents for this query
			if len(q_ids) == 0:
				q_ids += q_id

		if len(q_ids) < 1:
			return all_scores, all_q_ids
		scores = np.array(scores).flatten()
		tuples_of_doc_ids_and_scores = list(zip(d_ids, scores))
		sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse=True)


		if sorted_by_relevance is None:
			break

		all_scores.append(sorted_by_relevance)
		all_q_ids += q_id
	return all_scores, all_q_ids


# returns metric score if metric != None
# if metric = None returns scores, qiids

def test(model, mode, data_loaders, device, total_trained_samples, model_folder, reset=True, writer=None,
		 metric=None, report_top_N=-1):
	if isinstance(model, torch.nn.DataParallel):
		model_type = model.module.model_type
	else:
		model_type = model.model_type
	# if the model provides an indipendednt representation for the input (query/doc)
	if model_type == "representation-based":
		scores, q_ids = scores_representation_based(model, data_loaders[mode], device, writer,
													total_trained_samples, reset, model_folder,
													mode, plot=True)
	elif model_type == "interaction-based":
		scores, q_ids = scores_interaction_based(model, data_loaders[mode], device, reset)

	elif model_type == "rank-interaction":
		scores, q_ids = scores_interaction_based(model, data_loaders[mode], device, reset, classifier=True)

	elif model_type == "rank_prob":
		scores, q_ids = scores_all_comb(model, data_loaders[mode], device, reset, metric=metric)

	elif model_type == "bert-interaction":
		scores, q_ids = scores_bert_interaction(model, data_loaders[mode], device, reset)
	else:
		raise ValueError(f"run_model.py , model_type not properly defined!: {model_type}")

	# if requested, we only return the top N results for each query
	if report_top_N != -1:
		top_scores = []
		for i in range(len(q_ids)):
			top_scores.append(scores[i][:report_top_N])
		# scores[i] = scores[i][:report_top_N]
		scores = top_scores

	if metric:
		metric_score = metric.score(scores, q_ids)
		if writer:
			writer.add_scalar(f'test/{metric.name}', metric_score, total_trained_samples)
		print(f'{mode} -  {metric.name}: {metric_score}')
	else:
		metric_score = None

	return metric_score, scores, q_ids


def run(model, dataloaders, optim, loss_fn, epochs, writer, device, model_folder,
		l1_scalar=1, balance_scalar=1, patience=2, samples_per_epoch_train=10000, samples_per_epoch_val=10000,
		bottleneck_run=False, log_every_ratio=0.01, metric=None, validate=True, telegram=False, sub_batch_size=None):
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
	eval_mode = 'min' if validate else 'max'
	early_stopper = EarlyStopping(patience=patience, mode=eval_mode)
	total_trained_samples = 0

	# initialize data loader for the first epoch
	if total_trained_samples == 0:
		batch_iterator_train = iter(dataloaders['train'])
		if validate:
			batch_iterator_val = iter(dataloaders['val'])

	for epoch in range(1, epochs + 1):

		if early_stopper.stop:
			print(f"Early Stopping at Epoch: {epoch - 1}!")
			break

		print('Epoch', epoch)
		# training
		with torch.enable_grad():
			model.train()
			batch_iterator_train, total_trained_samples, train_total_loss, train_task_loss, train_l1_loss, train_l0_q, train_l0_docs, train_acc = run_epoch(
				model, 'train',
				dataloaders, batch_iterator_train, loss_fn, writer,
				l1_scalar, balance_scalar, total_trained_samples, device,
				optim=optim, samples_per_epoch=samples_per_epoch_train,
				log_every_ratio=log_every_ratio, sub_batch_size=sub_batch_size)



			telegram_message = f'Train:\nTotal loss {train_total_loss:.4f}\nTrain task_loss {train_task_loss:.4f}\nl1_loss {train_l1_loss:.4f}\nL0_query {train_l0_q:.4f}\nL0_docs {train_l0_docs:.4f}\nacc {train_acc:.4f}'
			telegram_message = model_folder + '\n' + telegram_message
			if telegram:
				subprocess.run(["bash", "telegram.sh", "-c", "-462467791", telegram_message])

			# in case the model has gone completely wrong, stop training
			if train_acc < 0.3:
				print('Ending training train because train accurracy is < 0.3!')
		# break

		# evaluation
		with torch.no_grad():
			model.eval()
			if bottleneck_run:
				print('Bottleneck run, stopping now!')
				break

			else:

				if validate:
					batch_iterator_val, _, val_total_loss, val_task_loss, val_l1_loss, val_l0_q, val_l0_docs, val_acc = run_epoch(model,
																																  'val',
																																  dataloaders,
																																  batch_iterator_val,
																																  loss_fn,
																																  writer,
																																  l1_scalar,
																																  balance_scalar,
																																  total_trained_samples,
																																  device,
																																  optim=None,
																																  samples_per_epoch=samples_per_epoch_val,
																																  log_every_ratio=log_every_ratio)

					if telegram:
						telegram_message = f'Validation:\nTotal loss {val_total_loss:.4f}\nTrain task_loss {val_task_loss:.4f}\nl1_loss {val_l1_loss:.4f}\nL0_query {val_l0_q:.4f}\nL0_docs {val_l0_docs:.4f}\nacc {val_acc:.4f}'
						telegram_message = model_folder + '\n' + telegram_message
						subprocess.run(["bash", "telegram.sh", "-c", "-462467791", telegram_message])

				# Run also proper evaluation script
				print('Running test: ')
				metric_score, scores, q_ids = test(model, 'test', dataloaders, device,
												   total_trained_samples, model_folder=model_folder, writer=writer,
												   metric=metric)
				# calculate mectric
				if telegram:
					telegram_message = model_folder + '\n' + f'Test Metric Score:\n{metric_score}'
					subprocess.run(["bash", "telegram.sh", "-c", "-462467791", telegram_message])

				if validate:
					metric_score = val_total_loss
				else:
					metric_score = metric_score

				# check for early stopping
				if not early_stopper.step(metric_score):
					print(f'Best model at current epoch {epoch}, av value: {metric_score}')
					# save best model so far to files
					torch.save(model.state_dict(), f'{model_folder}/best_model.model')
					metric.score(scores, q_ids, save_path=f'{model_folder}/rankings/best_model_epoch_{epoch}')

	return early_stopper.best, total_trained_samples
