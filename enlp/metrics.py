
import tempfile

from enlp.ms_marco_eval import compute_metrics_from_files

from enlp.trec_eval import TrecEval
from enlp.utils import write_ranking, write_ranking_trec


class Metric(object):

	def __init__(self, max_rank, qrel_file):
		self.name = None
		self.max_rank = max_rank
		tmp = tempfile.NamedTemporaryFile()
		self.tmp_ranking_file = tmp.name 
		self.qrel_file = qrel_file
	
	def write_scores(self, scores, qids):
		raise NotImplementedError()
	
	def score(self):
		raise NotImplementedError()


class MRR(Metric):
	def __init__(self, qrel_file, max_rank):
		super().__init__(max_rank, qrel_file)
		self.name = f'MRR@{max_rank}'	
	def write_scores(self, scores, qids):
		write_ranking(scores, qids, self.tmp_ranking_file)


	def score(self, scores, qids):
		self.write_scores(scores, qids)
		metric = compute_metrics_from_files(path_to_reference=self.qrel_file, path_to_candidate=self.tmp_ranking_file,
										  MaxMRRRank=self.max_rank)
		return round(metric, 6)
	
class MAPTrec(Metric):
	def __init__(self, trec_eval_path, qrel_file, max_rank, add_params=''):
		super().__init__(max_rank, qrel_file)
		self.name = f'MAP@{max_rank}'
		self.add_params = add_params
		self.trec_eval = TrecEval(trec_eval_path)
	def write_scores(self, scores, qids):
		write_ranking_trec(scores, qids, self.tmp_ranking_file)

	def score(self, scores, qids):
		print(scores)
		self.write_scores(scores, qids)
		metric = self.trec_eval.score(self.qrel_file, self.tmp_ranking_file, self.max_rank, self.add_params)
		return round(metric, 6)
