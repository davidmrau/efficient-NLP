import sys, os
from utils import *
os.environ["JAVA_HOME"] = "/Library/Java/JavaVirtualMachines/jdk-11.0.6.jdk/Contents/Home"


from pyserini.search import pysearch

folder = 'data/msmarco'

#collection = read_data(folder + '/collection.tsv')
query_id, queries = read_data(folder + '/queries.eval.small.tsv')

searcher = pysearch.SimpleSearcher('data/msmarco_index/')


for query in queries:

    hits = searcher.search(query)
    for i in range(0, 1000):
        print(f'{i+1} {hits[i].docid} {hits[i].score}')
