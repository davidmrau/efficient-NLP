
import argparse
import random
import numpy as np

from enlp.file_interface import FileInterface
from random import shuffle
import os
import time

class WeakTripletGenerator(object):
    def __init__(self, weak_results_filename, output_filename, top_k_per_query=1000, \
        sampler = 'uniform', samples_per_query = -1, min_results = 10, size_limit= None, queries_limit= None,
        train_val_ratio = 0.9, shuffle_queries = False):

        self.weak_results_fi = FileInterface(weak_results_filename)
        #self.queries = File(queries_filename)
        #self.documents = File(docs_filename)

        # override any previous files and open them

        self.output_filename_train = output_filename + "_train"
        self.output_filename_val = output_filename + "_val"
        self.output_file_train = open(self.output_filename_train, "w")
        self.output_file_val = open(self.output_filename_val, "w")

        #self.empty_doc_ids = set()

        self.size_limit = size_limit
        self.queries_limit = queries_limit

        self.queries_processed = 0

        self.min_results = min_results

        self.train_val_ratio = train_val_ratio

        self.top_k_per_query = top_k_per_query
        # defines the full(~1000) combinations to be calculated for a number of (samples_per_query) queries
        self.samples_per_query = samples_per_query

        # setting a maximum of 2000 candidates to sample from, if not specified differently from top_k_per_query
        self.max_candidates = top_k_per_query if top_k_per_query !=-1 else 2000

        if sampler == 'top_k':
            self.sampler_function = self.sample_top_k
        elif sampler == 'uniform':
            self.sampler_function = self.sample_uniform
        elif sampler == 'linear':
            self.sample_weights =  np.linspace(1,0,self.max_candidates)
            self.sampler_function = self.sample_linear
        elif sampler == 'zipf':
            # initialize common calculations
            self.sample_weights = np.asarray([1/(i+1) for i in range(self.max_candidates)])
            self.sampler_function = self.sample_zipf
        # top-N sampling methods, refer to uniform probability to the top N candidates and very small probability to the rest of the canidates
        elif "top-" in sampler:
            N = int(sampler.split('-')[1])
            self.sample_weights = np.asarray([1 for i in range(N)] + [0.0001 for i in range(self.max_candidates - N)])
            self.sampler_function = self.sample_top_k_probabilistically
        else:
            raise ValueError("Param 'sampler' of WeakSupervision, was not among {'top_k', 'uniform', 'zipf', 'linear', 'top-\{INTEGER}'}, but :" + str( sampler))
        # having a calculated list of indices, that will be used while sampling
        self.candidate_indices = list(range(self.max_candidates))

        # prepare query_index_list
        self.query_indices = list(range(len(self.weak_results_fi.seek_dict)))


        print("total Queries :", len(self.weak_results_fi.seek_dict))

        if shuffle_queries:
            shuffle(self.query_indices)



    def get_total_size_in_GB(self):
        def get_size_in_GB(filename):
            # get total bytes of file currently
            fileobject = open(filename, 'rb')
            fileobject.seek(0,2) # move the cursor to the end of the file
            size = fileobject.tell()
            fileobject.close()
            return float(size) / (1024*1024*1024)

        return get_size_in_GB(self.output_filename_train) + get_size_in_GB(self.output_filename_val)


    def check_termination_condition(self):

        # Translate in GBs
        current_file_sizes = self.get_total_size_in_GB()

        if (self.size_limit is not None) and (current_file_sizes > self.size_limit):
            print("Current file size:", current_file_sizes,", exceeds size limit:", self.size_limit,". Temrinating!")
            return True

        if (self.queries_limit is not None) and (self.queries_processed >= self.queries_limit):
            print("Current Number of processed Queries:", self.queries_processed,", exceeds limit:", self.queries_limit,". Temrinating!")
            return True

        return False




    def write_triplets(self):
        for query_index in self.query_indices:
            # split to training and validation queries -> triplets, according to provided ratio

            q_id, query_results = self.weak_results_fi.read_all_results_of_query_index(query_index, self.max_candidates)

            if len(query_results) < self.min_results:
                print("Insuficient number of results", len(query_results), ", Q_id", q_id)
                continue

            # if the content of the query is empty, then skip this query
            #query = self.queries[q_id]
            #if query is None:
            #    print("Query is None !",  q_id)
            #    continue

            self.queries_processed += 1

            # make sure that there are not any empty documents on the retrieved documents list (query_results)
            # since we are reading the documents we are also saving their contents in memory as an extra item in the final tupples
            non_empty_query_results_with_content = []
            for doc_id, score in query_results:
                # if it is not already known to be empty
                #if doc_id not in self.empty_doc_ids:

                    #document_content = self.documents[doc_id]
                    #if document_content is None:
                    #    self.empty_doc_ids.add(doc_id)
                    #    continue

                    # updating list with non empy_documents, and also adding the content of the document ot the tupple
                non_empty_query_results_with_content.append((doc_id, score))

            query_results = non_empty_query_results_with_content


            # in case we will end up using all the candidaes to create combinations
            if self.samples_per_query == -1 or len(query_results) <= self.samples_per_query :
                candidate_indices = [i for i in range( len(query_results) )]
            else:
                candidate_indices = self.sampler_function(scores_list = query_results, n = self.samples_per_query, return_indices = True)
                candidate_indices.sort()

            # generating a sample for each combination of i_th candidate with j_th candidate, without duplicates
            for i in candidate_indices:
                # if we do not request sampling, or there are not enough results to sample from, then we use all of them
                if (self.samples_per_query == -1) or (len(query_results) <= self.samples_per_query):
                    j_indices = list(range(len(query_results)))
                # otherwise we are able and requested to sample for the nested loop of combinations (j), so we do sample
                else:
                    j_indices = self.sampler_function(scores_list = query_results, n = self.samples_per_query, return_indices = True)

                for j in j_indices:
                    # making sure that we do not have any duplicates
                    if (j not in candidate_indices) or (j > i):

                        # doc at index i always has better score than doc at index j

                        doc1_id = query_results[i][0]
                        doc2_id = query_results[j][0]


                        if random.uniform(0,1) <= self.train_val_ratio:
                            output_file = self.output_file_train
                        else:
                            output_file = self.output_file_val

                        output_file.write(f"{q_id}\t{doc1_id}\t{doc2_id}\n")


            if self.check_termination_condition():
                break



    def shuf(self):
        print('Shuffling training and validation triples')
        os.system(f"cat {self.output_filename_train} | shuf -o {self.output_filename_train}")
        os.system(f"cat {self.output_filename_val} | shuf -o {self.output_filename_val}")

# sampling candidates functions
    def sample_uniform(self, scores_list, n, return_indices = False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        sampled_indices = np.random.choice(indices, size=n, replace=False)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]

    def sample_zipf(self, scores_list, n, return_indices = False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        # normalize sampling probabilities depending on the number of candidates
        p = self.sample_weights[:length] / sum(self.sample_weights[:length])
        sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]

    def sample_top_k(self, scores_list, n, return_indices = False):
        if return_indices:
            return [i for i in range(n)]
        return scores_list[:n]

    def sample_linear(self, scores_list, n, return_indices = False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        # normalize sampling probabilities depending on the number of candidates
        p = self.sample_weights[:length] / sum(self.sample_weights[:length])
        sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]

    # top-N sampling methods, refer to uniform probability to the top N candidates and very small probability to the rest of the canidates
    def sample_top_k_probabilistically(self, scores_list, n, return_indices = False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        # normalize sampling probabilities depending on the number of candidates
        p = self.sample_weights[:length] / sum(self.sample_weights[:length])
        sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]









parser = argparse.ArgumentParser()
parser.add_argument('--size_limit', type=float, default=None, help="Define max size of generated triplets file in GB.")
parser.add_argument('--queries_limit', type=int, default=None)
parser.add_argument('--sampler', type=str, default="top_k")
parser.add_argument('--target', type=str, default="binary")
parser.add_argument('--top_k_per_query', type=int, default=1000)
parser.add_argument('--samples_per_query', type=int, default=-1)
parser.add_argument('--min_hits', type=int, default=10)
parser.add_argument('--shuffle_queries', action='store_true')
# parser.add_argument('--shuffle_at_end', type=int, default=1000)

# parser.add_argument('--size_limit', type=float, default=None, help="Define max size of generated triplets file in GB.")

#parser.add_argument('--queries_filename', type=str, required=True)
#parser.add_argument('--docs_filename', type=str, required=True)
parser.add_argument('--weak_results_file', type=str, required=True)
args = parser.parse_args()


# args.weak_results_file = "/home/kondy/Desktop/Jaap/codes/LOCAL/efficient-NLP/data/robust04/robust04_AOL_anserini_top_1000_qld_no_stem_200k.filtered.debug.txt"
# args.queries_filename = "/home/kondy/Desktop/Jaap/codes/LOCAL/efficient-NLP/data/robust04/AOL-queries-all_filtered.txt.names_glove_stop_lucene_remove_unk_max_len_1500.tsv"
# args.docs_filename = "/home/kondy/Desktop/Jaap/codes/LOCAL/efficient-NLP/data/robust04/robust04_raw_docs.num_query_glove_stop_lucene_remove_unk_max_len_1500.tsv"
args.output_file = f"{args.weak_results_file}_TRIPLETS_{args.top_k_per_query}"
# args.shuffle = True

generator = WeakTripletGenerator(weak_results_filename = args.weak_results_file, output_filename = args.output_file, \
    top_k_per_query=args.top_k_per_query, sampler= args.sampler, min_results = args.min_hits, size_limit= args.size_limit,\
    samples_per_query = args.samples_per_query, queries_limit= args.queries_limit, shuffle_queries=args.shuffle_queries)


generator.write_triplets()




print("Triplet generation terminated! Total Queries processed:", generator.queries_processed)
print("Total Size in Gigabytes:", generator.get_total_size_in_GB())

#generator.shuf()
print("Don'f forget to shuffle the generated files!")
# in_fname = args.weak_results_file
# delimiter = args.delimiter
# out_fname = in_fname + '.offset_dict.p'


# offset_dict = create_seek_dictionary_per_1000_queries(in_fname, delimiter)

# p.dump(offset_dict, open(out_fname, 'wb'))
