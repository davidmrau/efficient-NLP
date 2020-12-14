import numpy as np
import collections
np.random.seed(2342)
def reprs_bert_interaction(model, dataloader, device, n_samples=5):

	attentions = []
	dataloader.reset()

	while True:
		count = 0
		# for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator_bert_interaction():
		q_attentions = collections.defaultdict(list)
		for q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids in dataloader.batch_generator_bert_interaction():
			# move data to device
			# propagate data through model
			out, attention = model(batch_input_ids, batch_attention_masks, batch_token_type_ids, output_attentions=True)
			for i, l in enumerate(attention):
				l = l.detach().cpu().numpy()
				for e in l:
					q_attentions[i].append(e)
			count += 1 
		if count == 0:
			break
		indices = range(len(q_attentions[0]))
		rand_ints = np.random.choice(indices, size=n_samples, replace=False)
		for k, v in q_attentions.items():
			print(len(q_attentions[k]))
			q_attentions[k] = q_attentions[k][rand_ints]	
		attentions.append(q_attentions)

	return attentions
