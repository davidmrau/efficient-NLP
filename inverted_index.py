import os
import shutil
import torch
from threading import Thread
from collections import defaultdict




# class for inverted index
class InvertedIndex:
    def __init__(self, path ,index_folder = "Inverted_Index", vocab_size = 100, num_of_workers = 4):

        self.path = f'{path}/{index_folder}'
        self.num_of_workers = num_of_workers
        self.vocab_size = vocab_size


    def initialize_index(self):
        """ Create an empty directory to save the index, and then initialize with one empty file for each dimension / posting list
        """
        # create index_folder, and delete existing (if there is one)
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)
        # create output directory
        os.makedirs(self.path)

        # create an empty file for each directory
        for i in range(self.vocab_size):
            open(os.path.join(self.path,str(i)) , 'a').close()


    def add_docs_to_index(self, doc_ids, activation_vectors):
        """ Given a batch of document id's and their sparse activations from a model,
            add the documents to the appropriate posting lists (multithreaded)
        """

        # number of non zero dimentions, over batch
        non_zero_dims = activation_vectors.sum(dim = 0).nonzero().squeeze().tolist()

        # maximum number of threads should be equal to total number of dimensions that have at least one non zero value
        num_of_workers = min(self.num_of_workers, len(non_zero_dims))

        # split the inputs into buckets
        buckets = []

        # for every thread
        for i in range(num_of_workers):
            # append template argument lists for the parameters (doc_ids, dim, activations)
            # doc_ids will be the same over all threads
            buckets.append([doc_ids, [], []])

        # for every non zero dimension
        for i in range(len(non_zero_dims)):
            # get thread index
            thread_index = i % num_of_workers
            # get paramters needed
            dim = non_zero_dims[i]
            batch_activations_on_that_dimension = activation_vectors[:, dim]
            # add them on thread's bucket parameters
            buckets[thread_index][1].append(dim)
            buckets[thread_index][2].append(batch_activations_on_that_dimension)

        # instanciate threads and execute them
        threads = []
        for i in range(num_of_workers):
            process = Thread(target = self.update_latent_terms_frequencies, args =  buckets[i])
            process.start()
            threads.append(process)

        # wait until all threads are finished
        for process in threads:
            process.join()


    def update_latent_terms_frequencies(self, doc_ids, dimensions, activations):
        """ Updates the posting lists of some Dimensions, for some Document ids
        Parameters
        ----------
        doc_ids     : list of (batch) document ids
        dimensions  : list of dimensions (on sparse space)
        activations : Model's output having the activations of all documents for these dimensions
        """

        for i, dim in enumerate(dimensions):
            # get indexes that have non zero activations for this dimension and need to be saved to the inverted index
            non_zero_indexes = activations[i].nonzero().squeeze().tolist()
            # in case of only one nonzero activation we get an integer, and we want to make turn it into a list
            if isinstance(non_zero_indexes, int):
                non_zero_indexes = [non_zero_indexes]

            # open the file that corresponds to this dimension
            posting_list_file = open( os.path.join(self.path, str(dim)) , "a")

            # append each necessaty doc_id and activation value
            for j in non_zero_indexes:
                doc_id = doc_ids[j]
                act_value = activations[i][j]
                posting_list_file.write(doc_id + "\t" + str(act_value.item()) + "\n")

            posting_list_file.close()


    def get_scores(self, query_ids, activation_vectors, top_results = -1, max_candidates_per_posting_list = -1):
        """ Given some query_ids, their sparse activation vectors
            return relevant documents sorted

        Parameters
        ----------
        query_ids                       : list
        activation_vectors              : Model's output
        top_results                     : top k results to return per query
        max_candidates_per_posting_list : top p documents to retrieve from each posting list (the posting lists are sorted)

        Returns
        -------
        results : [ (query_id1, [ (1st_doc_id, score), (2nd_doc_id, score), ... ] ),
                 (query_id2, [ (1st_doc_id, score), ... ] ), ... ]
                ]
        """
        # number of non zero dimentions, over batch
        non_zero_dims = activation_vectors.sum(dim = 0).nonzero().squeeze().tolist()

        # maximum number of threads should be equal to total number of dimensions that have at least one non zero value
        num_of_workers = min(self.num_of_workers, len(non_zero_dims))

        # split the inputs into buckets
        buckets = []

        # for every thread
        for i in range(num_of_workers):
            # append template argument lists for the parameters (doc_ids, dim, activations, doc_scores_dict)
            # query_ids will be the same over all threads
            # doc_scores_dict[query_id][doc_id] = score (for each thread separately)
            buckets.append([query_ids, [], [], max_candidates_per_posting_list, defaultdict(lambda: defaultdict(int))])

        # for every non zero dimension
        for i in range(len(non_zero_dims)):
            # get thread index
            thread_index = i % num_of_workers
            # get paramters needed
            dim = non_zero_dims[i]
            batch_activations_on_that_dimension = activation_vectors[:, dim]
            # add them on thread's bucket parameters
            buckets[thread_index][1].append(dim)
            buckets[thread_index][2].append(batch_activations_on_that_dimension)

        threads = []
        for i in range(num_of_workers):
            process = Thread(target = self.get_score_of_query_for_some_latent_terms, args =  buckets[i])
            process.start()
            threads.append(process)

        for process in threads:
            process.join()

        aggregated_scores = defaultdict(lambda: defaultdict(int))

        # aggregate scores
        for i in range(num_of_workers):
            scores_of_thread = buckets[i][-1]
            for query_id in scores_of_thread:
                for doc_id in scores_of_thread[query_id]:
                    aggregated_scores[query_id][doc_id] += scores_of_thread[query_id][doc_id]

        # sort the results and prepare them to be in the form [ (query_id1, [ (1st_doc_id, score), (2nd_doc_id, score)...]), (query_id2, [ (1st_doc_id, score), ... ] ), ... ]]
        results = []
        for query_id in aggregated_scores:
            query_results = [ (doc_id, aggregated_scores[query_id][doc_id]) for doc_id in aggregated_scores[query_id] ]
            sorted_results = sorted(query_results, key=lambda x: x[1], reverse = True)
            # retrieve only top top_results if specified
            if top_results != -1:
                sorted_results = sorted_results[:top_results]
            results.append((query_id, sorted_results))

        return results


    def get_score_of_query_for_some_latent_terms(self, query_ids, dimensions, activations, max_candidates_per_posting_list, scores_dict):
        """ Calculates the score of the query ids, for some posting lists (dimensions)

            Updates the scores_dict that is given as parameter, instead of returning an object
        """

        for i, dim in enumerate(dimensions):
            # get indexes that have non zero activations for this dimension and need to be saved to the inverted index
            non_zero_indexes = activations[i].nonzero().squeeze().tolist()
            # in case of only one nonzero activation we get an integer, and we want to make turn it into a list
            if isinstance(non_zero_indexes, int):
                non_zero_indexes = [non_zero_indexes]

            # open the file that corresponds to this dimension
            posting_list_file = open( os.path.join(self.path, str(dim)) , "r")

            line_counter = 0
            while(True):
                # read next line
                line = posting_list_file.readline().split("\t")
                # print(line[0], len(line))
                # check if the file has ended
                if len(line) < 2:
                    break
                # use only top max_candidates_per_posting_list candidates per posting list, if it is specified
                if max_candidates_per_posting_list != -1 and max_candidates_per_posting_list == line_counter:
                    break
                # retrieve doc_id and score
                doc_id = line[0]
                doc_activation = float(line[1])

                # update score of this doc, for every query that has activation in this dimension
                for j in non_zero_indexes:
                    query_id = query_ids[j]
                    query_activation = activations[i][j].item()
                    # inner product for this dimension
                    scores_dict[query_id][doc_id] += query_activation * doc_activation

                # increase the line counter
                line_counter += 1

            posting_list_file.close()



    def sort_posting_lists(self):
        """ Using Unix built-in "sort" function to sort all posting lists, with respect to the activation values
        """
        for i in range(self.vocab_size):
            filename =  os.path.join(self.path, str(i))
            os.system(f'sort -r -k 2 -o {filename} {filename}')


#
#
# num_of_docs = 100000
# dimensions = 10000
#
# ii = Inverted_Index(vocab_size = dimensions, num_of_workers=7)
#
# ii.initialize_index()
#
# docs = torch.rand(num_of_docs, dimensions)
# doc_ids = [ "doc_"+str(i) for i in range(num_of_docs)]
# docs = docs*(docs < 0.1).float()
#
#
# ii.add_docs_to_index(doc_ids, docs)
