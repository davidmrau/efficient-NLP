######
## creates an offset dict: {line index: byte-offset, ...} for efficient file loading
##
## example: offset_dict.py file_name
##
## saves the dict under file_name.offset_dict.p
#####

import sys
from utils import create_seek_dictionary_per_index
import pickle as p
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--delimiter', type=str, default=' ')
parser.add_argument('--fname', type=str)
parser.add_argument('--line_index_is_id', action='store_true')
args = parser.parse_args()


in_fname = args.fname
delimiter = args.delimiter
out_fname = in_fname + '.offset_dir.p'


offset_dict = create_seek_dictionary_per_index(in_fname, delimiter, args.line_index_is_id)

p.dump(offset_dict, open(out_fname, 'wb'))
