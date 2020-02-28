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

in_fname = sys.argv[1]
out_fname = in_fname + '.offset_dict.p'

offset_dict = create_seek_dictionary_per_index(in_fname)

p.dump(offset_dict, open(out_fname, 'wb'))
