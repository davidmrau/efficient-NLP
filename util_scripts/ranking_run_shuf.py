import argparse
from random import shuffle
parser = argparse.ArgumentParser()
parser.add_argument('--ranking', type=str, required=True)
args = parser.parse_args()

def write_ranking_trec(data, results_file_path):

	results_file = open(results_file_path, 'a+')
	for d in data:
		for i, k in enumerate(d):
			q_id, doc_id = k	
			results_file.write(f'{q_id}\tQ0\t{doc_id}\t{i}\t{len(d)-i}\teval\n')
	results_file.close()

prev_q_id = None
res = []
tmp_lines = []
f_out = f'{args.ranking}_shuf'

open(f_out, 'w').close()
with open(args.ranking, 'r') as rank_run:
	lines = rank_run.readlines()
	for line in lines:
		split = line.strip().split()
		q_id, doc_id  = split[0], split[2]

		if prev_q_id is None:
			prev_q_id = q_id

		if prev_q_id != q_id:
			print(len(tmp_lines))
			shuffle(tmp_lines)	
			res.append(tmp_lines)
			tmp_lines = [(q_id, doc_id)]
		else:	
			tmp_lines.append((q_id, doc_id))
		prev_q_id = q_id

res.append(tmp_lines)
write_ranking_trec(res, f_out)
