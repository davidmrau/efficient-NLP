
import tempfile

from enlp.ms_marco_eval import compute_metrics_from_files

from enlp.trec_eval import TrecEval
from enlp.utils import write_ranking, write_ranking_trec


class Metric(object):

	def __init__(self, max_rank, qrel_file, ranking_file_path=None):
		self.name = None
		self.max_rank = max_rank
		self.qrel_file = qrel_file
		if ranking_file_path:
			self.ranking_file_path = ranking_file_path
		else:
			tmp = tempfile.NamedTemporaryFile()
			self.ranking_file_path = tmp.name

	def write_scores(self, scores, qids, path):
		raise NotImplementedError()

	def score(self):
		raise NotImplementedError()


class MRR(Metric):
	def __init__(self, qrel_file, max_rank, ranking_file_path=None):
		super().__init__(max_rank, qrel_file, ranking_file_path)
		self.name = f'MRR@{max_rank}'
	def write_scores(self, scores, qids, path):
		write_ranking(scores, qids, path)


	def score(self, scores, qids):
		self.write_scores(scores, qids, self.ranking_file_path)
		metric = compute_metrics_from_files(path_to_reference=self.qrel_file, path_to_candidate=self.ranking_file_path,
										  MaxMRRRank=self.max_rank)
		return round(metric, 6)

class MAPTrec(Metric):
	def __init__(self, trec_eval_path, qrel_file, max_rank, add_params='', ranking_file_path=None):
		super().__init__(max_rank, qrel_file, ranking_file_path)
		self.name = f'MAP@{max_rank}'
		self.add_params = add_params
		self.trec_eval = TrecEval(trec_eval_path)

	def write_scores(self, scores, qids, path):
		write_ranking(scores, qids, f'{path}.tsv')
		write_ranking_trec(scores, qids, f'{path}.trec')

	def score(self, scores, qids, save_path=''):
		if save_path:
			path = save_path
		else:
			path = self.ranking_file_path

		self.write_scores(scores, qids, path)
		metric = self.trec_eval.score(self.qrel_file, f'{path}.trec', self.max_rank, self.add_params)
		return round(metric, 6)
