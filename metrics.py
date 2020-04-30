from ms_marco_eval import compute_metrics_from_files
import tempfile
from utils import write_ranking, write_ranking_trec
from trec_eval import TrecEval

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
		return compute_metrics_from_files(path_to_reference=self.qrel_file, path_to_candidate=self.tmp_ranking_file,
										  MaxMRRRank=self.max_rank)
	
class MAPTrec(Metric):
	def __init__(self, trec_eval_path, qrel_file, max_rank):
		super().__init__(max_rank, qrel_file)
		self.name = f'MAP@{max_rank}'
		self.trec_eval = TrecEval(trec_eval_path)
	def write_scores(self, scores, qids):
		write_ranking_trec(scores, qids, self.tmp_ranking_file)

	def score(self, scores, qids):
		self.write_scores(scores, qids)
		return self.trec_eval.score(self.qrel_file, self.tmp_ranking_file, self.max_rank)