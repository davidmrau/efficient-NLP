######
## creates an offset dict: {line index: byte-offset, ...} for efficient file loading
##
## example: offset_dict.py file_name
##
## saves the dict under file_name.offset_dict.p
#####

import argparse
import pickle as p


def create_seek_dictionary_per_index(filename, line_index_is_id = True):
	""" Creating a dictionary, for accessing directly a documents content, given document's id
			from a large file containing all documents
			returns:
			dictionary [doc_id] -> Seek value of a large file, so that you only have to read the exact document (doc_id)
	"""
	index_to_seek = {}
	sample_counter = 0

	with open(filename) as file:

		seek_value = file.tell()
		line = file.readline()
		while line:
			split_line = line.strip().split("\t")

			# triplets so use counter as id
			if line_index_is_id:
				id_ = sample_counter
			else:
				id_ = split_line[0].strip()

			sample_counter += 1
			index_to_seek[id_] = seek_value
			if sample_counter % 100000 == 0:
				print(sample_counter)
			seek_value = file.tell()
			line = file.readline()

	return index_to_seek



parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str)
parser.add_argument('--line_index_is_id', action='store_true')
args = parser.parse_args()


in_fname = args.fname
out_fname = in_fname + '.offset_dict.p'


offset_dict = create_seek_dictionary_per_index(in_fname, args.line_index_is_id)

p.dump(offset_dict, open(out_fname, 'wb'))
