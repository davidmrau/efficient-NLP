import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranking', type=str, required=True)
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()

ranking_run_dict ={}


with open(f'{args.ranking}_{args.rank}', 'w') as out_f:
	with open(args.ranking, 'r') as rank_run:
		lines = rank_run.readlines()
		for line in lines:
			split = line.strip().split()
			q_id, doc_id  = split[0], split[2]
			if q_id not in ranking_run_dict:
				ranking_run_dict[q_id] = [doc_id]
				out_f.write(line)			
			elif q_id in ranking_run_dict and len(ranking_run_dict[q_id]) < args.rank:
				ranking_run_dict[q_id].append(doc_id)
				out_f.write(line)			
