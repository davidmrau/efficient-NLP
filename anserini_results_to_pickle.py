######
## creates an offset dict: {line index: byte-offset, ...} for efficient file loading
##
## example: offset_dict.py file_name
##
## saves the dict under file_name.offset_dict.p
#####

import sys
import pickle as p
import argparse


def create_query_ids_list_and_their_results_dictionary(filename, delimiter=' '):
	""" reads anserini results and returns :
		-	a list of query ids in the order that they are written in the anserini results file
		-	a dictionary that has query indices as keys, and list of tuples in the form of (doc_id, score), as keys.
	"""

	q_idx_to_score_line_seek_dict = {}
	query_ids = []

	query_index = 0

	with open(filename) as file:

		prev_q_id = ""

		results = []

		# seek_value = file.tell()
		line = file.readline()

		while line:

			split_line = line.strip().split(delimiter)

			q_id = split_line[0].strip()


			# decompose line
			# split_line = line.split()

			# q_id = split_line[0]
			doc_id = split_line[2].strip()
			# rank = split_line[3]
			score = float(split_line[4])


			if q_id != prev_q_id:

				if len(results) > 0:
					query_ids.append(prev_q_id)


					q_idx_to_score_line_seek_dict[query_index - 1] = results

				results = [(doc_id, score)]

				prev_q_id = q_id
				query_index += 1

				if query_index % 1000 == 0:
					print(query_index)

			else:
				results.append((doc_id, score))

			# seek_value = file.tell()
			line = file.readline()

	query_ids.append(prev_q_id)
	q_idx_to_score_line_seek_dict[query_index - 1] = results


	return [query_ids, q_idx_to_score_line_seek_dict]


parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default=' ')
parser.add_argument('--fname', type=str)
args = parser.parse_args()


in_fname = args.fname
delimiter = args.delimiter
out_fname = in_fname + '.extracted_results.pickle'


extracted_results = create_query_ids_list_and_their_results_dictionary(in_fname, delimiter)


p.dump(extracted_results, open(out_fname, 'wb'))

# offset_dict = p.load(open(out_fname, 'rb'))

# query_ids, q_idx_to_score_line_seek_dict = offset_dict

# print(query_ids)

# for k in q_idx_to_score_line_seek_dict:
# 	print(k, query_ids[k])
# 	results = q_idx_to_score_line_seek_dict[k]
# 	for re in results:
# 		print(re)

# p.dump(offset_dict, open(out_fname, 'wb'))
