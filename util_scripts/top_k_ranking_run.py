import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ranking', type=str, required=True)
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()

prev_q_id = None
processed = 0

with open(f'{args.ranking}_{args.rank}', 'w') as out_f:
	with open(args.ranking, 'r') as rank_run:
		count = 0
		for line in rank_run:
			split = line.strip().split()
			q_id, doc_id  = split[0], split[2]
			if prev_q_id == None:
				prev_q_id = q_id
			if q_id != prev_q_id:
				count = 0
				processed += 1
				if processed % 1000 == 0:
					print(f'Processed: {processed}...')
			if count < args.rank:
				out_f.write(line)
			count += 1
			prev_q_id = q_id

