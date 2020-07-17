# converts trec style qrels to trec ranking file
# example usage:

#python3 qrel2ranking_result.py  qrel_filename



import sys

fname = sys.argv[1]

fname_pure = fname.split('/')[-1]
out_file = fname + '.strong_triples'
#AOL1 Q0 FR940810-0-00167 1 6.624100 Anserini
def make_line(q_id, doc_id, rank, score):
	return f'{q_id} Q0 {doc_id} {rank} {score} TrecQrel\n'


def write_query(res, out):
	res.sort(key=lambda x: x[2], reverse=True)
	rank = 0
	for el in res:
		rank += 1
		q_id, doc_id, score = el
		w_line = make_line(q_id, doc_id, rank, score)
		out.write(w_line)


prev_q_id = None

res = list()

with open(out_file, 'w') as out:
	with open(fname, 'r') as f:
		for line in f:
			# decompose line
			split_line = line.split()
			q_id = split_line[0].strip()
			doc_id = split_line[2]
			score = float(split_line[3])
			if prev_q_id == None:
				prev_q_id = q_id	
			if prev_q_id != q_id:
				res.append([q_id, doc_id, score])
				write_query(res, out)
				print(q_id)
				res = list()
				prev_q_id = q_id
			else:
				res.append([q_id, doc_id, score])
			
		write_query(res, out)
