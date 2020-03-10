import sys

fname = sys.argv[1]

f = open(fname, 'r')

line = f.readline()
fo = open(fname+'trec.txt', 'w')
while(line):
	qid, pid, rank = line.strip().split('\t')
	fo.write(f'{qid}\t0\t{pid}\t{rank}\t{1000-int(rank)}\teval\n')
	line = f.readline()

f.close()
fo.close()

