import subprocess

import sys


class TrecEval(object):
	def __init__(self, trec_eval_path):
		self.path = trec_eval_path

	def score(self, qrel_path, ranking_path, max_rank, add_params=''):

		#all topics
		output_all_topics = subprocess.check_output(f"./{self.path} {add_params} -q -m all_trec {qrel_path} {ranking_path} -M {max_rank}", shell=True).decode(sys.stdout.encoding)
		self.ranking_path = ranking_path
		all_topics_path = self.ranking_path + '.scores_all_topics.trec'
		print(all_topics_path)
		with open(all_topics_path, 'w') as f:
			f.write(output_all_topics)


		# overview trec_eval
		output = subprocess.check_output(f"./{self.path} {add_params} {qrel_path} {ranking_path} -M {max_rank}", shell=True).decode(sys.stdout.encoding)

		with open(self.ranking_path + '.scores.trec', 'w') as f:
			f.write(output)

		print(output)
		output = output.replace('\t', ' ').split('\n')
		for line in output:
			if line.startswith('map'):
				return float(line.split()[2])
