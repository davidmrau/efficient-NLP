import subprocess
import sys

class TrecEval(object):
	def __init__(self, trec_eval_path):
		self.path = trec_eval_path

	def score(self, qrel_path, ranking_path, max_rank):
		output = subprocess.check_output(f"./{self.path} {qrel_path} {ranking_path} -M  {max_rank}", shell=True).decode(sys.stdout.encoding)
		output = output.replace('\t', ' ').split('\n')
		for line in output:
			if line.startswith('map'):
				return float(line.split()[2])
		

