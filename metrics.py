from ms_marco_eval import compute_metrics_from_files
import TrecEval

class Metric(object):

    def __init__(self, name):
        self.name = name

    def score(self, qrel_file, ranking_file, max_rank):
        raise NotImplementedError()


class MRR(Metric):
    def __init__(self, name):
        super().__init__(name)

    def score(self, qrel_file, ranking_file, max_rank):
        return compute_metrics_from_files(path_to_reference=qrel_file, path_to_candidate=ranking_file,
                                          MaxMRRRank=max_rank)


class MAPTrec(Metric):
    def __init__(self, name, trec_eval_path):
        super().__init__(name)
        self.trec_eval = TrecEval(trec_eval_path)

    def score(self, qrel_file, ranking_file, max_rank):
        return self.trec_eval.score(qrel_file, ranking_file, max_rank)