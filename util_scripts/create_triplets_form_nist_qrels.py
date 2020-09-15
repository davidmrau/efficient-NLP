import argparse
import pickle
from transformers import BertTokenizerFast
import nltk
from nltk.stem import PorterStemmer
import pickle as p
import numpy as np
from collections import defaultdict

import os

def write_all_combinations_in_triplets(q_id, relevant, irrelevant, out_f):
	for rel_id in relevant:
		for irrel_id in irrelevant:
			out_f.write(q_id + '\t' + rel_id + '\t' + irrel_id + "\n")




# TREC official relevance scheme:

# Note: Documents were judged on a four-poiunt scale of 
# Not Relevant (0), Relevant (1), Highly Relevant (2) and Perfect (3). 
# Levels 1--3 are considered to be relevant for measures that use binary relevance judgments.
# Passages were judged on a four-point scale of Not Relevant (0), Related (1), Highly Relevant (2), and Perfect (3),
# where 'Related' is actually NOT Relevant---it means that the passage was on the same general topic,
# but did not answer the question. Thus, for Passage Ranking task runs (only), to compute evaluation
# measures that use binary relevance judgments using trec_eval, you either need to use
# trec_eval's -l option [trec_eval -l 2 qrelsfile runfile] or modify the qrels file to change all 1 judgments to 0.


def generate_triplets(args):


	ending = "_pointwise" if args.pointwise else "_pairwise"


	out_f_name = args.qrels_file + ending
	if args.binary_pairwise:
		out_f_name += '_binary'
	if args.rand_negative_docs_dict:
		out_f_name += '_rand_docs'

	qrels = defaultdict(lambda: defaultdict(list))

	# arg.pointwise defines whether the generated triplets will be used for 
	# pairiwse training or pointwise training

	# if the qrels will be used for pointwise training
	# then each document-query pair should be used only once

	# if the qrels are going to be used for pairwise training,
	# then we can reuse the same relevant document to create triplets
	# with many irrelevant documets 


	for line in open(args.qrels_file, 'r'):
		line = line.split()

		q_id = line[0]
		d_id = line[2]
		score = int(line[3])

		# if score != 0 and score != 1 and score != 2:
		# 	print(line)


		qrels[q_id][score].append( d_id )

	# qrels do not contain enough irrelevant documents,
	# so we will add some from the top-1000 that are not classified as relevant

	if args.top_1000_file != 'none':

		for line in open(args.top_1000_file, 'r'):

			line = line.split()
			q_id = line[0]
			d_id = line[2]

			# if we haven't read this document id already from the qrels file
			if d_id not in qrels[q_id][0] and d_id not in qrels[q_id][1] and d_id not in qrels[q_id][2] and d_id not in qrels[q_id][3]:
				qrels[q_id][-1].append(d_id)

	
	out_f = open(out_f_name,  'w')
	for q_id in qrels:
		perfectly_relevant = qrels[q_id][3]
		highly_relevant = qrels[q_id][2]
		related = qrels[q_id][1]
		irrelevant = qrels[q_id][0]
		irrelevant_top_1000 = qrels[q_id][-1]

		if args.pointwise:
			relevant = perfectly_relevant + highly_relevant

			irrelevant = related + irrelevant + irrelevant_top_1000

			num_of_triplets = len(relevant)

			irrelevant = irrelevant[ : num_of_triplets]

			if len(relevant) > len(irrelevant):
				print("Irreleants are less than relevants !! ")
				print("Q_id:",q_id, ", Rel:", len(relevant), ", Irrel:", len(irrelevant))

			for rel_id, irrel_id in zip(relevant, irrelevant):

				out_f.write(q_id + '\t' + rel_id + '\t' + irrel_id + "\n")
		elif args.binary_pairwise:
			relevant = perfectly_relevant + highly_relevant + related 
			write_all_combinations_in_triplets(q_id, relevant, irrelevant, out_f)
		elif args.rand_negative_docs_dict is not None:
			all_docs = p.load(open(args.rand_negative_docs_dict, 'rb'))
			relevant = perfectly_relevant + highly_relevant + related 
			num_of_triplets = len(relevant)
			irrelevant = np.random.choice(list(all_docs), num_of_triplets, replace=False)
			print(irrelevant)
			

			write_all_combinations_in_triplets(q_id, relevant, irrelevant, out_f)

		else:
			# generate pairs using all relevance hierarchy combinations:

			# perfectly_relevant - highly_relevant
			write_all_combinations_in_triplets(q_id, perfectly_relevant, highly_relevant, out_f)
			# perfectly_relevant - related
			write_all_combinations_in_triplets(q_id, perfectly_relevant, related, out_f)
			# perfectly_relevant - irrelevant
			write_all_combinations_in_triplets(q_id, perfectly_relevant, irrelevant, out_f)
			# perfectly_relevant - irrelevant_top_1000
			write_all_combinations_in_triplets(q_id, perfectly_relevant, irrelevant_top_1000, out_f)

			# highly_relevant - related
			write_all_combinations_in_triplets(q_id, highly_relevant, related, out_f)
			# highly_relevant - irrelevant
			write_all_combinations_in_triplets(q_id, highly_relevant, irrelevant, out_f)
			# highly_relevant - irrelevant_top_1000
			write_all_combinations_in_triplets(q_id, highly_relevant, irrelevant_top_1000, out_f)

			# related - irrelevant
			write_all_combinations_in_triplets(q_id, related, irrelevant, out_f)
			# related - irrelevant_top_1000
			write_all_combinations_in_triplets(q_id, related, irrelevant_top_1000, out_f)

			# irrelevant - irrelevant_top_1000
			write_all_combinations_in_triplets(q_id, irrelevant, irrelevant_top_1000, out_f)



if __name__ == "__main__":


	parser = argparse.ArgumentParser()
	parser.add_argument('--qrels_file', type=str)
	parser.add_argument('--binary_pairwise', action='store_true')
	parser.add_argument('--rand_negative_docs_dict', default=None, type=str)
	parser.add_argument('--top_1000_file', type=str, default='none')
	parser.add_argument('--pointwise', action='store_true')
	args = parser.parse_args()
	    
	for arg in vars(args):
		print(arg, ':', getattr(args, arg))
	# args.qrels_file = "/home/kondy/Desktop/Jaap/codes/LOCAL/efficient-NLP/data/msmarco/2019qrels-pass.txt"

	# args.top_1000_file = "/home/kondy/Desktop/Jaap/codes/LOCAL/efficient-NLP/data/msmarco/msmarco-passagetest2019-top1000_43_ranking_results_style.tsv"
	
	# pointwise generates around 2.5 k triplets

	generate_triplets(args)
