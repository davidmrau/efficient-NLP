import numpy as np

np.random.seed(2342)
def reprs_bert_interaction(model, dataloader, device, n_sample=5):

	attentions = []
	dataloader.reset()

	while True:
		count = 0
		# for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator_bert_interaction():
		q_attentions = list()
		for q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids in dataloader.batch_generator_bert_interaction():
			# move data to device
			# propagate data through model
			out, attention = model(batch_input_ids, batch_attention_masks, batch_token_type_ids, output_attentions=True)
			q_attentions.append([l.detach().cpu().numpy() for l in attention])
			count += 1
		rand_ints = np.random.randint(0, high=len(q_attentions), n_sample)
		print(rand_ints)
		q_attentions = q_attentions[rand_ints]	
		q_attentions = np.concatenate(q_attentions, 1)
		print(q_attentions.shape)
		attentions.append(q_attentions)
		if count == 0:
			break
	return attentions
