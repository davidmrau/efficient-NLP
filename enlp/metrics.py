
import tempfile

from enlp.ms_marco_eval import compute_metrics_from_files

from enlp.trec_eval import TrecEval
from enlp.utils import write_ranking, write_ranking_trec


class Metric(object):

	def __init__(self, max_rank, qrel_file, file_path=None):
		self.name = None
		self.max_rank = max_rank
		tmp = tempfile.NamedTemporaryFile()
		self.tmp_ranking_file = tmp.name 
		self.qrel_file = qrel_file
		if file_path:
			self.file_path = file_path + '/ranking'
		else:
			self.file_path = file_path
	
	def write_scores(self, scores, qids):
		raise NotImplementedError()
	
	def score(self):
		raise NotImplementedError()


class MRR(Metric):
	def __init__(self, qrel_file, max_rank, file_path=None):
		super().__init__(max_rank, qrel_file, file_path)
		self.name = f'MRR@{max_rank}'	
	def write_scores(self, scores, qids):
		write_ranking(scores, qids, self.tmp_ranking_file)
		if self.file_path:
			write_ranking(scores, qids, self.file_path)


	def score(self, scores, qids):
		self.write_scores(scores, qids)
		metric = compute_metrics_from_files(path_to_reference=self.qrel_file, path_to_candidate=self.tmp_ranking_file,
										  MaxMRRRank=self.max_rank)
		return round(metric, 6)
	
class MAPTrec(Metric):
	def __init__(self, trec_eval_path, qrel_file, max_rank, add_params='', save_all_path=None):
		super().__init__(max_rank, qrel_file, save_all_path)
		self.name = f'MAP@{max_rank}'
		self.add_params = add_params
		if save_all_path:
			save_all_path += 'all_results.trec'
		else:
			save_all_path = None
		self.trec_eval = TrecEval(trec_eval_path, save_all_path)
	def write_scores(self, scores, qids):
		write_ranking_trec(scores, qids, self.tmp_ranking_file)
		if self.file_path:
			write_ranking(scores, qids, self.file_path + '.tsv')
			write_ranking_trec(scores, qids, self.file_path + '.trec')

	def score(self, scores, qids):
		self.write_scores(scores, qids)
		metric = self.trec_eval.score(self.qrel_file, self.tmp_ranking_file, self.max_rank, self.add_params)
		return round(metric, 6)
