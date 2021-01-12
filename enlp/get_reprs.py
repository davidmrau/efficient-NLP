import numpy as np
import collections
import sys
import pickle

def reprs_bert_interaction(model, dataloader, device, path, output_attentions=False, output_hidden_states=False):
	attentions, hidden_states = None, None
	for batch in dataloader:
		batch = [item.to(device) for item in batch]
		input_ids, attention_masks, token_type_ids = batch 
		# propagate data through model
		#out, attention = model(input_ids, attention_masks, token_type_ids, output_attentions=attentions, output_hidden_states=hidden_states)
		out, outputs = model(input_ids, attention_masks, token_type_ids, output_attentions=output_attentions, output_hidden_states=output_hidden_states)
		if output_attentions:
			attention = outputs.attentions	
			if not attentions:
				attentions = [[] for i in range(len(attention))]
			for i, l in enumerate(attention):
				l = l.detach().cpu().numpy()
				attentions[i].append(l) 
		if output_hidden_states:	
			hidden_state = outputs.hidden_states
			if not hidden_states:
				hidden_states = [[] for i in range(len(hidden_state))]
			for i, l in enumerate(hidden_state):
				l = l.detach().cpu().numpy()
				hidden_states[i].append(l)
	if hidden_states: 
		pickle.dump(hidden_states, open(f'{path}/hidden_states.p', 'wb'))
	if attentions:
		pickle.dump(attentions, open(f'{path}/attentions.p', 'wb'))

def reprs_bert_interaction_ranking_results(model, dataloader, device, fname,  n_samples=10):
	dataloader.reset()
	num_q = 0
	while True:
		count = 0
		print(num_q)
		q_attentions = None
		for q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids in dataloader.batch_generator_bert_interaction():
			# move data to device
			# propagate data through model
			out, attention = model(batch_input_ids, batch_attention_masks, batch_token_type_ids, output_attentions=True)
			if not q_attentions:
				q_attentions = [[] for i in range(len(attentions))]
			for i, l in enumerate(attention):
				l = l.detach().cpu()
				#for j, e in enumerate(l):
				#	length_input = sum(batch_attention_masks[j])
				#	e = e[:,:length_input, :length_input]
				#	q_attentions[i].append(e)
				q_attentions[i].append(l) 
			count += 1
		if count == 0:
			break
		else:
			#num_attentions = len(q_attentions[0])
			#n_samples = min(n_samples, num_attentions)
			#rand_ints = np.random.choice(range(num_attentions), size=n_samples, replace=False)
			#print(rand_ints)
			#for k, v in q_attentions.items():	
			#	q_attentions[k] = [v[i] for i in rand_ints]
			#attentions.append(q_attentions)
			pickle.dump(q_attentions, open(f'{fname}_{num_q}.p', 'wb'))
			num_q += 1
