import argparse



parser = argparse.ArgumentParser(description='Given a ranking_run file re-rank only top {rank} hits with respect to ranking_ref.')
parser.add_argument('--ranking_run', type=str, required=True)
parser.add_argument('--ranking_ref', type=str, required=True)
parser.add_argument('--rank', type=int, required=True)
args = parser.parse_args()

ranking_run_dict = {}
with open(args.ranking_ref, 'r') as rank_ref:
    lines = rank_ref.readlines()
    for line in lines:
        split = line.strip().split()
        q_id, doc_id = split[0], split[2]
        if len(ranking_run_dict[q_id]) > args.rank:
            ranking_run_dict.setdefault(q_id, []).append((doc_id, line))

res = {}
with open(f'{args.ranking_run}_{args.rank}', 'w') as out_f:
    with open(args.ranking_run, 'r') as rank_run:
        lines = rank_run.readlines()
        for line in lines:
            split = line.strip().split()
            q_id, doc_id = split[0], split[2]

            #if # docs of query <= args.rank:
            if len(res[q_id]) <= args.rank:
                # if document exists later in the ranking but is now in top {rank}, delete from later ranking
                if doc_id in map(lambda x: x[0], ranking_run_dict[q_id]):
                    res.setdefault(q_id, []).append(line)
                    ranking_run_dict[q_id].remove(doc_id)
            else:
                # take line from ranking_run_dict
                query_lines = res[q_id] + list(map(lambda x: x[1], ranking_run_dict[q_id]))

                for l in query_lines:
                    out_f.write(l)
