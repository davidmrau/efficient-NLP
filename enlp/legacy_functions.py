
class WeakSupervision(IterableDataset):
    def __init__(self, weak_results_fi, queries_fi, documents_fi, top_k_per_query=-1, sampler='uniform',
                 target='binary',
                 samples_per_query=-1, single_sample=False, shuffle=True, min_results=2, strong_negatives=True,
                 indices_to_use=None,
                 sample_j=False, sample_random=False, max_length=-1):

        # "open" triplets file
        self.weak_results_file = weak_results_fi

        # "open" documents file
        self.documents = documents_fi

        # create a list of docment ids
        self.doc_ids_list = list(self.documents.seek_dict)

        # "open" queries file
        self.queries = queries_fi

        self.top_k_per_query = top_k_per_query
        # defines the full(~1000) combinations to be calculated for a number of (samples_per_query) queries
        self.samples_per_query = samples_per_query

        # if True, then we create exactly one positive sample for each query
        self.single_sample = single_sample
        # if strong_negatives == True then then reassuring that negative samples are not among the (weakly) relevant ones
        self.strong_negatives = strong_negatives

        self.shuffle = shuffle
        # if sample_j is True, then we sample samples_per_query samples for creating the cpmbinations. sample different ones for each (i)
        self.sample_j = sample_j

        self.min_results = min_results

        if target == 'binary':
            self.target_function = self.binary_target
        elif target == 'rank_prob':
            self.target_function = self.probability_difference_target
        else:
            raise ValueError(
                "Param 'target' of WeakSupervision, was not among {'binary', 'rank_prob'}, but :" + str(target))

        # setting a maximum of 2000 candidates to sample from, if not specified differently from top_k_per_query
        self.max_candidates = top_k_per_query if top_k_per_query != -1 else 2000
        if sampler == 'top_n':
            self.sampler_function = self.sample_top_n
        elif sampler == 'uniform':
            self.sampler_function = self.sample_uniform
        elif sampler == 'linear':
            self.sample_weights = np.linspace(1, 0, self.max_candidates)
            self.sampler_function = self.sample_linear
        elif sampler == 'zipf':
            # initialize common calculations
            self.sample_weights = np.asarray([1 / (i + 1) for i in range(self.max_candidates)])
            self.sampler_function = self.sample_zipf
        # top-N sampling methods, refer to uniform probability to the top N candidates and very small probability to the rest of the canidates
        elif "top-" in sampler:
            N = int(sampler.split('-')[1])
            self.sample_weights = np.asarray([1 for i in range(N)] + [0.0001 for i in range(self.max_candidates - N)])
            self.sampler_function = self.sample_top_n_probabilistically
        else:
            raise ValueError(
                "Param 'sampler' of WeakSupervision, was not among {'top_n', 'uniform', 'zipf', 'linear', 'top-\{INTEGER}'}, but :" + str(
                    sampler))
        # having a calculated list of indices, that will be used while sampling
        self.candidate_indices = list(range(self.max_candidates))

        # this will be used later in case suffle is True
        if indices_to_use is None:
            self.query_indices = list(range(len(self.weak_results_file)))
        else:
            self.query_indices = indices_to_use

        # whether to also generate triplets using a relevand and a random strong negative document from corpus
        self.sample_random = sample_random

        self.max_length = max_length

    def __len__(self):
        raise NotImplementedError()

    def generate_triplet(self, query, candidates):
        # shuffle order of candidates
        random.shuffle(candidates)
        # get target
        target = self.target_function(candidates[0], candidates[1])
        # get document content from tupples
        doc1 = candidates[0][2]
        doc2 = candidates[1][2]

        if self.max_length != -1:
            query = query[:self.max_length]
            doc1 = doc1[:self.max_length]
            doc2 = doc2[:self.max_length]

        return [query, doc1, doc2], target

    def __iter__(self):

        if self.shuffle:
            random.shuffle(self.query_indices)

        for q_index in self.query_indices:
            q_id, query_results = self.weak_results_file.read_all_results_of_query_index(q_index, self.top_k_per_query)

            # if the content of the query is empty, then skip this query
            query = self.queries[q_id]
            if query is None:
                continue

            # skip queries that do not have the necessary nuber of results
            if len(query_results) < self.min_results:
                continue

            # reassuring that negative results are not among the weak scorer results, by creting a set with all relevant ids
            if self.strong_negatives:
                relevant_doc_ids_set = {doc_id for doc_id, _ in query_results}

            #  if we are generating exactly one relevant sample for each query (and one negative)
            if self.single_sample:
                # sample candidates
                candidate_indices = self.sampler_function(scores_list=query_results, n=2, return_indices=True)

                doc1_id, score1 = query_results[candidate_indices[0]]
                doc2_id, score2 = query_results[candidate_indices[1]]

                doc1 = self.documents[doc1_id]
                doc2 = self.documents[doc2_id]

                candidates = [(doc1_id, score1, doc1), (doc2_id, score2, doc2)]
                # print(q_id, doc1_id, doc2_id)

                if (doc1 is not None) and (doc2 is not None):
                    # yield triplet of relevants
                    yield self.generate_triplet(query, candidates)

                else:
                    continue

                if self.sample_random:

                    # get the first of the candidates in order to be matched with a random negative document
                    result1 = candidates[0]

                    # add the relevant document id to the excluding list if we haven't already
                    if self.strong_negatives == False:
                        rel_doc_id = result1[0]
                        relevant_doc_ids_set = {rel_doc_id}

                    negative_result = self.sample_negative_document_result(exclude_doc_ids_set=relevant_doc_ids_set)

                    yield self.generate_triplet(query, [result1, negative_result])

            #  if we are generating all combinations from samples_per_query candidates with all the candidates samples
            # (plus 1 negative sample for each of the afforementioned samples)
            else:

                # make sure that there are not any empty documents on the retrieved documents list (query_results)
                # since we are reading the documents we are also saving their contents in memory as an extra item in the final tupples
                non_empty_query_results_with_content = []
                for doc_id, score in query_results:
                    document_content = self.documents[doc_id]
                    if document_content is not None:
                        # updating list with non empy_documents, and also adding the content of the document ot the tupple
                        non_empty_query_results_with_content.append((doc_id, score, document_content))
                query_results = non_empty_query_results_with_content

                # inb case we will end up using all the candidaes to create combinations
                if self.samples_per_query == -1 or len(query_results) <= self.samples_per_query:
                    candidate_indices = [i for i in range(len(query_results))]
                else:
                    candidate_indices = self.sampler_function(scores_list=query_results, n=self.samples_per_query,
                                                              return_indices=True)
                    candidate_indices.sort()

                # generating a sample for each combination of i_th candidate with j_th candidate, without duplicates
                for i in candidate_indices:
                    # if we do not request sampling, or there are not enough results to sample from, then we use all of them
                    if (self.sample_j == False) or (self.samples_per_query == -1) or (
                            len(query_results) <= self.samples_per_query):
                        j_indices = list(range(len(query_results)))
                    # otherwise we are able and requested to sample for the nested loop of combinations (j), so we do sample
                    else:
                        j_indices = self.sampler_function(scores_list=query_results, n=self.samples_per_query,
                                                          return_indices=True)

                    for j in j_indices:
                        # making sure that we do not have any duplicates
                        if (j not in candidate_indices) or (j > i):

                            # yield triplet of relevants
                            candidate1 = query_results[i]
                            candidate2 = query_results[j]
                            yield self.generate_triplet(query, [candidate1, candidate2])

                            if self.sample_random:
                                # yield triplet of irrelevants
                                # add the relevant document id to the excluding list if we haven't already
                                if self.strong_negatives == False:
                                    rel_doc_id = candidate1[0]
                                    relevant_doc_ids_set = {rel_doc_id}

                                negative_result = self.sample_negative_document_result(
                                    exclude_doc_ids_set=relevant_doc_ids_set)

                                yield self.generate_triplet(query, [candidate1, negative_result])

    # target value calculation functions
    # binary targets -1/1 defining which is the more relevant candidate out of the two candidates
    def binary_target(self, result1, result2):
        # 1 if result1 is better and -1 if result2 is better
        target = 1 if result1[1] > result2[1] else -1
        return target

    # implementation of the rank_prob model's target from paper : Neural Ranking Models with Weak Supervision (https://arxiv.org/abs/1704.08803)
    def probability_difference_target(self, result1, result2):
        target = result1[1] / (result1[1] + result2[1])
        return target

    # sampling candidates functions
    # sample a negative candidate from the collection
    def sample_negative_document_result(self, exclude_doc_ids_set):
        # get a random index from documents' list
        random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
        # get the corresponding document id
        random_doc_id = self.doc_ids_list[random_doc_index]
        # retrieve content of the random document
        document_content = self.documents[random_doc_id]

        # make sure that the random document's id is not in the exclude list and its content is not empty
        while random_doc_id in exclude_doc_ids_set or document_content is None:
            # get a random index from documents' list
            random_doc_index = random.randint(0, len(self.doc_ids_list) - 1)
            # get the corresponding document id
            random_doc_id = self.doc_ids_list[random_doc_index]
            # retrieve content of the random document
            document_content = self.documents[random_doc_id]

        return (random_doc_id, 0, document_content)

    # sampling out of relevant documents functions :

    def sample_uniform(self, scores_list, n, return_indices=False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        sampled_indices = random.sample(indices, n)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]

    def sample_zipf(self, scores_list, n, return_indices=False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        # normalize sampling probabilities depending on the number of candidates
        p = self.sample_weights[:length] / sum(self.sample_weights[:length])
        sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]

    def sample_top_n(self, scores_list, n, return_indices=False):
        if return_indices:
            return [i for i in range(n)]
        return scores_list[:n]

    def sample_linear(self, scores_list, n, return_indices=False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        # normalize sampling probabilities depending on the number of candidates
        p = self.sample_weights[:length] / sum(self.sample_weights[:length])
        sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]

    # top-N sampling methods, refer to uniform probability to the top N candidates and very small probability to the rest of the canidates
    def sample_top_n_probabilistically(self, scores_list, n, return_indices=False):
        length = len(scores_list)
        indices = self.candidate_indices[:length]
        # normalize sampling probabilities depending on the number of candidates
        p = self.sample_weights[:length] / sum(self.sample_weights[:length])
        sampled_indices = np.random.choice(indices, size=n, replace=False, p=p)
        if return_indices:
            return sampled_indices
        return [scores_list[i] for i in sampled_indices]


def split_batch_to_minibatches(batch, max_samples_per_gpu = 2, n_gpu = 1):

    if max_samples_per_gpu == -1:
        return [batch]

    data, targets, lengths = batch

    # calculate the number of minibatches so that the maximum number of samples per gpu is maintained
    size_of_minibatch = max_samples_per_gpu * n_gpu

    split_size = data.size(0) // 3
    queries, doc1, doc2 = torch.split(data, split_size)
    queries_len, doc1_len, doc2_len = torch.split(lengths, split_size)

    number_of_samples_in_batch = queries.size(0)

    if number_of_samples_in_batch <= max_samples_per_gpu:
        return [batch]


    number_of_minibatches = math.ceil(number_of_samples_in_batch / size_of_minibatch)

    # arrange the minibatches
    minibatches = []
    for i in range(number_of_minibatches):

        minibatch_queries =  queries[ i * size_of_minibatch : (i+1) * size_of_minibatch ]

        minibatch_d1 =  doc1[  i * size_of_minibatch : (i+1) * size_of_minibatch ]
        minibatch_d2 =  doc2[  i * size_of_minibatch : (i+1) * size_of_minibatch ]

        minibatch_queries_lengths =  queries_len[ i * size_of_minibatch : (i+1) * size_of_minibatch ]

        minibatch_d1_lengths =  doc1_len[  i * size_of_minibatch : (i+1) * size_of_minibatch ]
        minibatch_d2_lengths =  doc2_len[  i * size_of_minibatch : (i+1) * size_of_minibatch ]

        minibatch_targets =  targets[ i * size_of_minibatch : (i+1) * size_of_minibatch ]

        minibatch_data = torch.cat([minibatch_queries , minibatch_d1 , minibatch_d2], dim = 0)
        minibatch_lengths = torch.cat([minibatch_queries_lengths , minibatch_d1_lengths , minibatch_d2_lengths], dim = 0)

        minibatch = [minibatch_data, minibatch_targets, minibatch_lengths]

        minibatches.append(minibatch)
    return minibatches



def split_batch_to_minibatches_bert_interaction(batch, max_samples_per_gpu = 2, n_gpu = 1, pairwise_training = False):

    # calculate the number of minibatches so that the maximum number of samples per gpu is maintained
    size_of_minibatch = max_samples_per_gpu * n_gpu

    batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_targets = batch

    number_of_samples_in_batch = batch_input_ids.size(0)

    if number_of_samples_in_batch <= max_samples_per_gpu:

        if pairwise_training:
            # rows represent sample pairs, column 1 and 2 represent target of d1 and d2 respecttively
            batch_targets = batch_targets.view(-1,2)
            # rows represent sample pairs, 1 means d1 is more relevant, -1 means d2 is more relevant
            batch_targets = (batch_targets[:,0] > batch_targets[:,1]).int()*2 -1

            batch = [batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_targets]

        return [batch]

    number_of_minibatches = math.ceil(number_of_samples_in_batch / size_of_minibatch)

    minibatches = []

    if pairwise_training == False:

        for i in range(number_of_minibatches):
            minibatches.append([ batch_input_ids[i*size_of_minibatch : (i+1)*size_of_minibatch], \
                                 batch_attention_masks[i*size_of_minibatch : (i+1)*size_of_minibatch], \
                                 batch_token_type_ids[i*size_of_minibatch : (i+1)*size_of_minibatch], \
                                 batch_targets[i*size_of_minibatch : (i+1)*size_of_minibatch] ])

    else:

        # preprocess targets so that they prepresent pair targets

        # rows represent sample pairs, column 1 and 2 represent target of d1 and d2 respecttively
        batch_targets = batch_targets.view(-1,2)
        # rows represent sample pairs, 1 means d1 is more relevant, -1 means d2 is more relevant
        batch_targets = (batch_targets[:,0] > batch_targets[:,1]).int()*2 -1
        # in a pairwise training case, the number of targets are half of the size of the input samples
        size_of_minibatch_targets = int(size_of_minibatch/2)

        for i in range(number_of_minibatches):
            minibatches.append([ batch_input_ids[i*size_of_minibatch : (i+1)*size_of_minibatch], \
                                 batch_attention_masks[i*size_of_minibatch : (i+1)*size_of_minibatch], \
                                 batch_token_type_ids[i*size_of_minibatch : (i+1)*size_of_minibatch], \
                                 batch_targets[i*size_of_minibatch_targets : (i+1)*size_of_minibatch_targets] ])

    return minibatches