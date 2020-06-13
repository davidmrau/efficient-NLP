######
## creates an offset dict: {line index: byte-offset, ...} for efficient file loading
##
## example: offset_dict.py file_name
##
## saves the dict under file_name.offset_dict.p
#####

import argparse
import pickle as p


def create_seek_dictionary_per_result_structured_by_query_index(filename, delimiter=' '):
	""" Creating a dictionary, for accessing directly a anserini results content for a specific query index
			from the large anserini results file. Query index is the number of queries that preceed this query on
			on the anserini results file
			returns:
			dictionary [doc_id] -> Seek value of a large file, so that we can start reading the results file 
									from the seek value that this query's result start
	"""

	q_idx_to_score_line_seek_dict = {}

	query_index = 0

	with open(filename) as file:

		prev_q_id = ""

		results_seek_values = []

		seek_value = file.tell()
		line = file.readline()

		while line:

			split_line = line.strip().split(delimiter)

			q_id = split_line[0].strip()

			if q_id != prev_q_id:

				if len(results_seek_values) > 0:
					q_idx_to_score_line_seek_dict[query_index - 1] = results_seek_values

				results_seek_values = [seek_value]

				prev_q_id = q_id
				query_index += 1

				if query_index % 1000 == 0:
					print(query_index)

			else:
				results_seek_values.append(seek_value)

			seek_value = file.tell()
			line = file.readline()

	q_idx_to_score_line_seek_dict[query_index - 1] = results_seek_values


	return q_idx_to_score_line_seek_dict


parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default=' ')
parser.add_argument('--fname', type=str)
args = parser.parse_args()


in_fname = args.fname
delimiter = args.delimiter
out_fname = in_fname + '._detailed_anserini_results_offset_dict.p'


offset_dict = create_seek_dictionary_per_result_structured_by_query_index(in_fname, delimiter)

# print(offset_dict)

p.dump(offset_dict, open(out_fname, 'wb'))
