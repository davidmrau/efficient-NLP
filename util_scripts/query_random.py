import argparse
import random
from collections import defaultdict
parser = argparse.ArgumentParser(description='Given a file in the form \'id\ttoken_1 token_n\' randomly shuffle tokens.')
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()


lines = open(args.file, 'r').read().splitlines()
with open(args.file + '_random', 'w') as out:

	for line in lines:
		id_, token = line.split('\t')
		tokens = token.split()
		random.shuffle(tokens)	
		out.write(f'{id_}\t{" ".join(tokens)}\n')	
