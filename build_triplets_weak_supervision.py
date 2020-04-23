from omegaconf import OmegaConf


def write_to_triplets(q_id, scores_list, triplets_file):
	for doc_id_1, score_1 in scores_list:
		for doc_id_2, score_2 in scores_list:
			if doc_id_1 != doc_id_2:
				triplets_file.write("{}\t{}\t{}\t{:.4f}\n".format(q_id, doc_id_1, doc_id_2, score_1 / (score_2 + score_1)))


def build_triplets(galago_results_filename, triplets_filename, keep_top_k):


	triplets_file = open(triplets_filename + "_top_" + str(keep_top_k), 'w')


	with open(galago_results_filename) as file:


		# prev_q_id = None

		# q_id = 0

		prev_q_id = None


		line = file.readline()

		while line:

		# for line in fp:
			split_line = line.split()

			q_id = split_line[0]
			doc_id = split_line[2]
			# rank = split_line[3]
			score = float(split_line[4])

			# if we started reading another query
			if q_id != prev_q_id:

				# if this is not the first query we are reading
				if prev_q_id is not None:
					write_to_triplets(q_id = prev_q_id, scores_list = scores_list, triplets_file = triplets_file)

				scores_list = []

			prev_q_id = q_id

			line = file.readline()

			# only using the keep_top_k relevant docs for each query
			if len(scores_list) != keep_top_k:
				# add this relevant document with its score to the list of relevant documents for this query
				scores_list.append((doc_id, score))


	write_to_triplets(q_id = prev_q_id, scores_list = scores_list, triplets_file = triplets_file)


	triplets_file.close()





if __name__ == "__main__":
	# getting command line arguments
	cl_cfg = OmegaConf.from_cli()
	# settting up default values
	if not cl_cfg.galago_results_filename :
		cl_cfg.galago_results_filename = "test_result_trec"
	if not cl_cfg.triplets_filename :
		cl_cfg.triplets_filename = "triplets"
	if not cl_cfg.keep_top_k :
		cl_cfg.keep_top_k = 1000



	build_triplets(galago_results_filename = cl_cfg.galago_results_filename, triplets_filename = cl_cfg.triplets_filename, keep_top_k = cl_cfg.keep_top_k)

