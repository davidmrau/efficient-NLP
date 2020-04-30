import subprocess


class TrecEval(object):
    def __init__(self, trec_eval_path):
        self.path = trec_eval_path

    def score(self, qrel_path, ranking_path, max_rank):
        output = subprocess.check_output(f"./{self.path} {qrel_path} {ranking_path} -M  {max_rank}", shell=True)
        print(output)


trec_eval = TrecEval('trec_eval')
trec_eval.score('sdf', 'g', 100)
