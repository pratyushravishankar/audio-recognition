
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import utils
import features as ft
import time
import lsh_random_projection as LSH
import resource

import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as

features = utils.load('data/fma_metadata/features.csv')
tracks = utils.load('data/fma_metadata/tracks.csv')


class Evaluation:

    # pass in lsh table instances for spectral, probe, ...
    def __init__(self, lsh):
        self.lsh = lsh

    # lsh_probe_1
    # lsh_probe_2
    # spectral/
    # lsh_ranodmised_proj

    # def eval_genre_accuracy(self):

    #     # eval same lsh instance with genre accuracy lsh_probe 1 & 2, spectral_hashing

    #     # create lsh instance, with populated dataset
    #     # try 10 different queries
    #     # time them for probe methods + wihtouth probe

    #     # create spectral_hashing isinstance
    #     # time for spectral_hashing

    #     lsh = LSH.LSH(1, 5, 140)

    #     queries = getQueries()

    #     time_list = get_list_time(lsh, queries)

    # def get_list_times(querymethod,  queries):
    #     None

        # for q in queries:
        # tic = time.perf_counter()
        # lsh.get(q)
        # toc = time.perf_counter()
        # time.list.append(toc - tic)

    def get_boxplot_rand_projection(self, X):

        # print(X)

        ys = []
        xs = []

        # TODO compile all data from 100 queries into same array.

        for i in range(10):

            ratio = (i + 1) / 10

            ys.append(ratio)

            # for row in X.iterrows():
            # print(">> > ", row)
            query_df = X.iloc[1:2]

            # print(query_df)

            matches = lsh.get(query_df, ratio, probeType="rand-proj")
            print("ratio: ", ratio, "ROW : ", matches)

            xs.append(matches)

            # print(matches)

        # ax = matplotlib.pyplot.boxplot(xs, ys)

        # print(ax)
#
        # print("XS ", xs)
        # print("YS ", ys)

        # print(">>>>> I " i)
        # print(matches)`

    def get_recall_accuracy(self, X):
        # TODO
        # get with 100 queries
        # matches_list = getquries()
        matches_list = lsh.get(X, probeType="rand-proj")

        brute_forces = eval.bruteforce_get(features['mfcc'], X)
        avg_recall = 0
        count = 0

        for matches, answers in zip(matches_list, brute_forces):

            print("MATCHES ", matches)

            print("ANSWERS ", answers)

            #     # webrute_forces_list = []

            # for idx, ys in enumerate(brute_forces_list):

            recall = self.get_search_quality(matches['id'], answers['id'])

            print("RATIO >>> ", recall)

            avg_recall = avg_recall + recall
            count = count + 1

        return avg_recall / count

    def eval_top_k_accuracy(self):

        print("starting eval")

        # query_df = ft.compute_features(query)
        query_df = features.iloc[1:2]

        brute_force_top_k = self.bruteforce_get(
            features['mfcc'], query_df['mfcc'])

        lsh_random_proj_top_k = self.lsh.get(
            query_df['mfcc'], probeType="rand_proj")

        lsh_probe_step_wise_top_k = self.lsh.get(
            query_df['mfcc'], probeType="step-wise")

        # TODO modularise lsh code so acc working

        lsh_probe_bit_flip_top_k = self.lsh.get(
            query_df['mfcc'], probeType="bit-flip")
        # spectral_top_k =

        lsh_random_proj_score = self.get_search_quality(
            brute_force_top_k['id'], lsh_random_proj_top_k['id'])

        lsh_probe_step_wise_score = self.get_search_quality(
            brute_force_top_k['id'], lsh_probe_step_wise_top_k['id'])

        lsh_probe_bit_flip_score = self.get_search_quality(
            brute_force_top_k['id'], lsh_probe_bit_flip_top_k['id'])

        print("randproj: ", lsh_random_proj_score, " step-wise: ",
              lsh_probe_step_wise_score, " bit_flip ", lsh_probe_bit_flip_score)

        print("BRUETY ", brute_force_top_k)
        print("RAND PROJ ", lsh_random_proj_top_k)
        print("STEP-wise ", lsh_probe_step_wise_top_k)
        print("bit flip ", lsh_probe_bit_flip_top_k)

        # spectral_top_k_score =

    def get_search_quality(self, ys, Ys):

        k = len(ys)
        if k == 0:
            return 0

        # print("STRAT")

        count = 0
        for Y in Ys:
            if (ys == Y).any():

                # print("FOUND ", Y)
                count = count + 1

        return count / k

    def bruteforce_get(self, features, inp_vec, k=20):

        query_top_ks = [None for i in range(len(inp_vec))]

        for idx in range(len(inp_vec)):

            distance = pairwise_distances(
                features, inp_vec.iloc[idx], metric='euclidean').flatten()

            nearest_neighbours = pd.DataFrame({'id': features.index, 'genre': tracks['track']['genre_top'], 'distance': distance}).sort_values(
                'distance').reset_index(drop=True)

            # print("nearest negih")
            # print(nearest_neighbours.head())

            candidate_set_labels = nearest_neighbours.sort_values(
                by=['distance'], ascending=True)

            non_null = candidate_set_labels[candidate_set_labels['genre'].notnull(
            )]

            query_top_ks[idx] = non_null.iloc[:k]

        return query_top_ks


lsh = LSH.LSH(1, 25, 140)
# lsh.add(features['mfcc'])


# eval.eval_top_k_accuracy()
# query_df = features.iloc[1:2]
# brute_force_top_k = eval.bruteforce_get(
#     features['mfcc'], query_df['mfcc'])


# print("Brute-force : ", brute_force_top_k)

# val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# print("Process usage: ", val)

X_train, X_test = train_test_split(features, test_size=2)

lsh.add(X_train['mfcc'])
eval = Evaluation(lsh)

res = eval.get_recall_accuracy(X_test['mfcc'])

print("TOTAL RATIO ", res)
