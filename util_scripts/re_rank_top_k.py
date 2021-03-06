import argparse
from collections import defaultdict
from statistics import mean 
import numpy as np

parser = argparse.ArgumentParser(description='Given a ranking_run file re-rank top {rank} hits with respect to ranking_ref.')
parser.add_argument('--ranking_run', type=str, required=True)
parser.add_argument('--ranking_ref', type=str, required=True)
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()

ranking_ref = defaultdict(list)
ranking_run = defaultdict(list)
res = defaultdict(list)
count_ranked_higher = defaultdict(lambda: 0)

def write_ranking_trec(q_ids, results_file_path):
    with open(results_file_path, 'w') as results_file:
        for q_id in q_ids:
            max_rank = len(q_ids[q_id])
            for j, doc_id in enumerate(q_ids[q_id]):
                # print(j, line)
                res = f'{q_id}\tQ0\t{doc_id}\t{j}\t{max_rank-j}\teval\n'
                results_file.write(res)

with open(args.ranking_run, 'r') as rank_run:
    lines = rank_run.readlines()
    for line in lines:
        split = line.strip().split()
        q_id, doc_id = split[0], split[2]
        if len(ranking_run[q_id]) < args.rank:
            ranking_run[q_id].append(doc_id)


with open(args.ranking_ref, 'r') as rank_ref:
    lines = rank_ref.readlines()
    for line in lines:
        split = line.strip().split()
        q_id, doc_id = split[0], split[2]
        
        if doc_id not in ranking_run[q_id]:
                 ranking_ref[q_id].append(doc_id)
        else:
            ranking_ref[q_id].append(None)
        if len(ranking_run[q_id]) == 0:
             del ranking_run[q_id]

for q_id in ranking_run.keys():
    for e in ranking_run[q_id]:
        res[q_id].append(e)

    for e in ranking_ref[q_id][len(ranking_run[q_id]):]:
        if e is not None:
            res[q_id].append(e)
write_ranking_trec(res, f'{args.ranking_run}_{args.rank}')

print(f'wrote file to {args.ranking_run}_{args.rank}')
print(f'number of docs per query that differ between run and ref ranking in the top-{args.rank}:')
av_new_docs, av_brought_up = list(), list()
for k, v in ranking_ref.items():
    if len(ranking_run[k]) == 0:
        continue
    # None means in top-k means that doc_id exists already in top-k ranking run and therefore inverse is # of new docs
    new_docs = len(ranking_run[k]) - sum([1 for e in v[:args.rank] if e is None])
    av_new_docs.append(new_docs)
    # None after top-k means that exists already in top-k ranking run and therefore was brought up from below in ranking_ref 
    brought_up = sum([1 for e in v[args.rank+1:] if e is None])
    av_brought_up.append(brought_up)
    print(f'{k}: \n\tnew docs within top-{args.rank}: {new_docs}\n\tdocs brought up to top-{args.rank}: {brought_up}')

print(f'Average: \n\tnew docs within top-{args.rank} {mean(av_new_docs):.2f}\n\tdocs brought up to top-{args.rank}: {mean(av_brought_up):.2f}')
