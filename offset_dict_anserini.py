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


def create_seek_dictionary_per_1000_queries(filename, delimiter=' '):
	""" Creating a dictionary, for accessing directly a anserini results content for a specific query index
			from the large anserini results file. Query index is the number of queries that preceed this query on
			on the anserini results file
			returns:
			dictionary [doc_id] -> Seek value of a large file, so that we can start reading the results file 
									from the seek value that this query's result start
	"""
	index_to_seek = {}
	sample_counter = 0

	with open(filename) as file:

		prev_q_id = ""

		seek_value = file.tell()
		line = file.readline()
		while line:
			split_line = line.strip().split(delimiter)

			q_id = split_line[0].strip()

			if q_id != prev_q_id:
				index_to_seek[sample_counter] = seek_value
				prev_q_id = q_id
				sample_counter += 1

				if sample_counter % 1000 == 0:
					print(sample_counter)

			seek_value = file.tell()
			line = file.readline()

	return index_to_seek




parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default=' ')
parser.add_argument('--fname', type=str)
args = parser.parse_args()


in_fname = args.fname
delimiter = args.delimiter
out_fname = in_fname + '.offset_dict.p'


offset_dict = create_seek_dictionary_per_1000_queries(in_fname, delimiter)

p.dump(offset_dict, open(out_fname, 'wb'))
