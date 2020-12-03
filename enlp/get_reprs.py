


def reprs_bert_interaction(model, dataloader, device):

	attentions = []
	dataloader.reset()

	while True:

		# for q_id, data_q, length_q, batch_ids_d, batch_data_d, batch_lengths_d in dataloader.batch_generator_bert_interaction():
		for q_id, d_batch_ids, batch_input_ids, batch_attention_masks, batch_token_type_ids in dataloader.batch_generator_bert_interaction():
			# move data to device
			batch_input_ids, batch_attention_masks, batch_token_type_ids = batch_input_ids.to(
				device), batch_attention_masks.to(device), batch_token_type_ids.to(device)
			# propagate data through model
			_, attention = model(batch_input_ids, batch_attention_masks, batch_token_type_ids)
			attentions.append(attentiont)
			if batch_input_ids is None:
				break
	return attentions
