import torch
from utils import l1_loss_fn, l0_loss_fn, balance_loss_fn, write_ranking, l0_loss
import os
from utils import plot_histogram_of_latent_terms, plot_ordered_posting_lists_lengths


def log_progress(writer, mode, total_trained_samples, currently_trained_samples, samples_per_epoch, loss, l1_loss, balance_loss, total_loss, l0_q, l0_docs, acc):
	print("  {}/{} task loss: {:.4f}, l1 loss: {:.4f}, balance loss: {:.4f}".format(currently_trained_samples, samples_per_epoch, loss, l1_loss, balance_loss))
		# update tensorboard
	writer.add_scalar(f'{mode}_task_loss', loss, total_trained_samples  )
	writer.add_scalar(f'{mode}_l1_loss', l1_loss, total_trained_samples)
	writer.add_scalar(f'{mode}_balance_loss', balance_loss, total_trained_samples)
	writer.add_scalar(f'{mode}_total_loss', total_loss, total_trained_samples)
	writer.add_scalar(f'{mode}_L0_query', l0_q, total_trained_samples)
	writer.add_scalar(f'{mode}_L0_docs', l0_docs, total_trained_samples)
	writer.add_scalar(f'{mode}_acc', acc, total_trained_samples)


def run_epoch(model, mode, dataloader, batch_iterator, loss_fn, epoch, writer, l1_scalar, balance_scalar, total_trained_samples, device, optim=None, samples_per_epoch = 10000, log_every_ratio = 0.01):
	"""Train 1 epoch, and evaluate every 1000 total_training_steps. Tensorboard is updated after every batch

	Returns
	-------
	av_loss                 : (float) average loss
	total_trained_samples    : (int) the total number of training steps performed over all epochs
	type
		Description of returned object.
	"""
	log_every_ratio = max(dataloader[mode].batch_size/samples_per_epoch,   log_every_ratio)
	prev_trained_samples = total_trained_samples

	current_trained_samples = prev_trained_samples - total_trained_samples

	current_log_threshold = log_every_ratio

	cur_trained_samples = 0


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

		# decompose batch
		data, targets, lengths = batch

		# get number of samples within the batch
		batch_samples_number = targets.size(0)

		# update the number of trained samples in this epoch
		cur_trained_samples += batch_samples_number
		# update the total number of trained samples
		total_trained_samples += batch_samples_number

		# forward pass (inputs are concatenated in the form [q1, q2, ..., q1d1, q2d1, ..., q1d2, q2d2, ...])
		logits = model(data.to(device), lengths.to(device))
		# moving targets also to the appropriate device
		targets = targets.to(device)

		# accordingly splitting the model's output for the batch into triplet form (queries, document1 and document2)
		split_size = logits.size(0)//3
		q_repr, d1_repr, d2_repr = torch.split(logits, split_size)

		# performing inner products
		dot_q_d1 = torch.bmm(q_repr.unsqueeze(1), d1_repr.unsqueeze(-1)).squeeze()
		dot_q_d2 = torch.bmm(q_repr.unsqueeze(1), d2_repr.unsqueeze(-1)).squeeze()

		# calculating loss
		loss = loss_fn(dot_q_d1, dot_q_d2, targets)

		# calculate l1 loss
		l1_loss = l1_loss_fn(torch.cat([q_repr, d1_repr, d2_repr], 1))
		# calculate balance loss
		balance_loss = balance_loss_fn(logits, device)
		# calculating L0 loss
		l0_q, l0_docs = l0_loss_fn(d1_repr, d2_repr, q_repr)

		# calculating classification accuracy (whether the correct document was classified as more relevant)
		acc = (((dot_q_d1 > dot_q_d2).float() == targets).float()+ ((dot_q_d2 > dot_q_d1).float() == targets*-1).float()).mean()

		# aggregating losses and running backward pass and update step
		total_loss = loss +  l1_loss * l1_scalar + balance_loss * balance_scalar
		# if we are training, then we perform the backward pass and update step
		if optim != None:
			optim.zero_grad()
			total_loss.backward()
			optim.step()

		torch.cuda.empty_cache()


		# get pogress ratio
		samples_trained_ratio = cur_trained_samples / samples_per_epoch

		# check whether we should log
		if samples_trained_ratio > current_log_threshold:
			# log
			log_progress(writer, mode, total_trained_samples, cur_trained_samples, samples_per_epoch, loss, l1_loss, balance_loss, total_loss, l0_q, l0_docs, acc)
			# update log threshold
			current_log_threshold += log_every_ratio

	# log the values of the final training step
	log_progress(writer, mode, total_trained_samples, cur_trained_samples, samples_per_epoch, loss, l1_loss, balance_loss, total_loss, l0_q, l0_docs, acc)



	return total_trained_samples


def get_all_reprs(model, dataloader, device):
	with torch.no_grad():
		model.eval()
		reprs = list()
		ids = list()
		for batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator():
			repr_ = model(batch_data_d.to(device), batch_lengths_d.to(device))
			reprs.append(repr_)
			ids += batch_ids_d
		return reprs, ids

def get_scores(doc_reprs, doc_ids, q_reprs, max_rank):
	scores = list()
	for batch_q_repr in q_reprs:
		batch_len = len(batch_q_repr)
		# q_score_lists = [ []]*batch_len
		q_score_lists = [[] for i in range(batch_len) ]
		for batch_doc_repr in doc_reprs:
			dots_q_d = batch_q_repr @ batch_doc_repr.T
			# appending scores of batch_documents for this batch of queries
			for i in range(batch_len):
				q_score_lists[i] += dots_q_d[i].detach().cpu().tolist()


		# now we will sort the documents by relevance, for each query
		for i in range(batch_len):
			tuples_of_doc_ids_and_scores = [(doc_id, score) for doc_id, score in zip(doc_ids, q_score_lists[i])]
			sorted_by_relevance = sorted(tuples_of_doc_ids_and_scores, key=lambda x: x[1], reverse = True)
			if max_rank != -1:
				sorted_by_relevance = sorted_by_relevance[:max_rank]
			scores.append(sorted_by_relevance)
	return scores


def evaluate(model, mode,data_loaders, device, max_rank, writer,  total_trained_samples, metric, reset=True):

	query_batch_generator, docs_batch_generator = data_loaders

	if reset:
		docs_batch_generator.reset()
		query_batch_generator.reset()

	d_repr, d_ids = get_all_reprs(model, docs_batch_generator, device)
	q_repr, q_ids = get_all_reprs(model, query_batch_generator, device)
	scores = get_scores(d_repr, d_ids, q_repr, max_rank)


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


	writer.add_scalar(f'{mode}_l1_loss', l1_loss, total_trained_samples)

	writer.add_scalar(f'{mode}_L0_query', q_l0_loss, total_trained_samples)
	writer.add_scalar(f'{mode}_L0_docs', d_l0_loss, total_trained_samples)

	metric_score = metric.score(scores, q_ids)

	writer.add_scalar(f'{metric.name}', metric_score, total_trained_samples)
	print(f'{mode} -  {metric.name}: {metric_score}')

	return scores, q_repr, d_repr, q_ids, d_ids, metric_score


def train(model, dataloaders, optim, loss_fn, epochs, writer, device, model_folder, sparse_dimensions, metric, max_rank=1000,
			l1_scalar = 1, balance_scalar= 1, patience = 2, samples_per_epoch = 10000, debug = False, bottleneck_run = False, log_every_ratio = 0.01):
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
	best_metric_score = -1
	temp_patience = 0
	total_trained_samples = 0


	# initialize data loader for the first epoch
	if total_trained_samples == 0:
		batch_iterator = iter(dataloaders['train'])

	for epoch in range(1, epochs+1):
		print('Epoch', epoch)
		# training
		with torch.enable_grad():
			model.train()
			total_trained_samples = run_epoch(model, 'train', dataloaders, batch_iterator, loss_fn, epoch, writer, l1_scalar, balance_scalar, total_trained_samples, device,
							optim=optim, samples_per_epoch = samples_per_epoch)
		# evaluation
		with torch.no_grad():
			model.eval()

			if bottleneck_run:
				metric_score = 0.001

			else:
				# run ms marco eval
				scores, q_repr, d_repr, q_ids, _, metric_score = evaluate(model,'val', dataloaders, device, writer,total_trained_samples,metric,  max_rank=max_rank)

				# check for early stopping
				if metric_score > best_metric_score:
					print(f'Best model at current epoch {epoch}, {metric.name}: {metric_score}')
					temp_patience = 0
					best_metric_score = metric_score
					# save best model so far to file
					torch.save(model, f'{model_folder}/best_model.model' )

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

				if metric_score < 0.03 and not debug:
					print(f"{metric.name} smaller than 0.03. Ending Training!")
					break


	if not bottleneck_run:
		# load best model
		model = torch.load(f'{model_folder}/best_model.model')
	


	return model, metric_score
