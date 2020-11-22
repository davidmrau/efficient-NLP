import argparse

import numpy as np
from collections import defaultdict
parser = argparse.ArgumentParser(description='Given a ranking_run file re-rank only top {rank} hits with respect to ranking_ref.')
parser.add_argument('--ranking_run', type=str, required=True)
parser.add_argument('--ranking_ref', type=str, required=True)
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()


def write_ranking_trec(q_ids, results_file_path):
    with open(results_file_path, 'w') as results_file:
        for q_id in q_ids:
            max_rank = len(q_ids[q_id])
            for j, (doc_id, line) in enumerate(q_ids[q_id]):
                # print(j, line)
                results_file.write(f'{q_id}\tQ0\t{doc_id}\t{j}\t{max_rank-j}\teval\n')
def read2dict(fname):
    res = defaultdict(lambda: list())
    with open(fname, 'r') as f:
        for line in f:
            split = line.strip().split()
            q_id, doc_id = split[0], split[2]
            res[q_id].append(line)


    return res

ref = read2dict(args.ranking_ref)
run = read2dict(args.ranking_run)
res = defaultdict(lambda: list())
written = defaultdict(lambda: list())
count_ranked_higher =  defaultdict(lambda: 0)



for k in ref.keys():
    for i in range(len(ref[k])):


        split_ref = ref[k][i].strip().split()
        q_id_ref, doc_id_ref, rank_ref = split_ref[0], split_ref[2], split_ref[3]


        if i < len(run[k]):


            split_run = run[k][i].strip().split()
            q_id_run, doc_id_run, rank_run = split_run[0], split_run[2], split_run[3]


            #if # docs of query <= args.rank:
            if len(written[q_id_run]) < args.rank:
                written[q_id_run].append(doc_id_run)
                res[q_id_run].append((doc_id_run, 'run '+run[k][i]))

            else:
                if doc_id_ref not in written[q_id_ref]:
                    res[q_id_ref].append((doc_id_ref, 'ref '+ref[k][i]))
                else:
                    count_ranked_higher[k] += 1
        else:
                if doc_id_ref not in written[q_id_ref]:
                    res[q_id_ref].append((doc_id_ref, 'ref '+ref[k][i]))
                else:
                    count_ranked_higher[k] += 1


write_ranking_trec(res, f'{args.ranking_run}_rerank_top_{args.rank}')


print(f'number of docs per query that differ between run and ref ranking in the top {args.rank}:')
for k, v in count_ranked_higher.items():
    print(f'{k}: {v}')
print(f'av: {np.mean(list(map(int, count_ranked_higher.values()))):.2f}')
