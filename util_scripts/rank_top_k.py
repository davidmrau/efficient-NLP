import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranking_run', type=str, required=True)
parser.add_argument('--ranking_ref', type=str, required=True)
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()

ranking_run_dict ={}


with open(args.ranking_ref, 'r') as rank_ref:
	lines = rank_ref.readlines()
	for line in lines:
		split = line.strip().split()
		q_id, doc_id  = split[0], split[2]
		if q_id not in ranking_run_dict:
			ranking_run_dict[q_id] = [doc_id]
		elif q_id in ranking_run_dict and len(ranking_run_dict[q_id]) < args.rank:
			ranking_run_dict[q_id].append(doc_id)
with open(f'{args.ranking_run}_{args.rank}', 'w') as out_f:
	with open(args.ranking_run, 'r') as rank_run:
		lines = rank_run.readlines()
		for line in lines:
			split = line.strip().split()
			q_id, doc_id  = split[0], split[2]
			if q_id in ranking_run_dict and doc_id in ranking_run_dict[q_id]:
				out_f.write(line)			
