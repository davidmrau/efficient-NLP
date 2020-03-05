
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str, default='data/msmarco/msmarco-passagetest2019-top1000.tsv')


cfg = parser.parse_args()

fname = cfg.fname
with open(fname, 'r') as f_in:
	with open(fname+'.d_id_doc.tsv', 'w') as f_out:
		line = f_in.readline()
		prev_id = ''
		while line:
			splt_line = line.split('\t')
			id_, text = splt_line[1], splt_line[3]
			# since the same doc can be relevant for different queries, we filter out document duplicates
			if id_ != prev_id:
				f_out.write(id_ + '\t' + text)
			prev_id = id_
			line = f_in.readline()
