
# import matplotlib
from tqdm import tqdm
import librosa
from scipy import stats
import warnings
import multiprocessing
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import utils
import features as ft
import time
import lsh_random_projection as LSH
import resource
import numpy as np

import os

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
        # tic = time.perf_counter()        # lsh.get(q)
        # toc = time.perf_counter()
        # time.list.append(toc - tic)

    def grid_search(self):

        key_sizes = [i for i in range(5, 40, 5)]
        tables_sizes = [i for i in range(1, 50)]

        for key in key_sizes:
            for table in tables_sizes:

                # def get_boxplot_rand_projection(self, X):

                #     # print(X)

                #     ys = []
                #     xs = []

                #     # # TODO compile all data from 100 queries into same array.

                #     for i in range(10):

                #         ratio = (i + 1) / 10

                #         ys.append(ratio)

                #         matches = lsh.get(inp_vec=X, collision_ratio=i,
                #                           probeType="rand_proj")

                #     #     # for row in X.iterrows():
                #     #     # print(">> > ", row)
                #     #     query_df = X.iloc[1:2]

                #     #     # print(query_df)

                #     #     # matches = lsh.get(query_df, ratio, probeType="rand-proj")
                #     #     # print("ratio: ", ratio, "ROW : ", matches)

                #     #     xs.append(matches)

                #     #     # print(matches)

                #     # plt.boxplot(xs, ys)

                #     # plt.show()

        np.random.seed(19680801)

        # fake up some data
        spread = np.random.rand(50) * 100
        center = np.ones(25) * 50
        flier_high = np.random.rand(10) * 100 + 100
        flier_low = np.random.rand(10) * -100
        data = np.concatenate((spread, center, flier_high, flier_low))
        fig1, ax1 = plt.subplots()
        ax1.set_title('Basic Plot')
        ax1.boxplot(data)
        plt.show()

#
        # print("XS ", xs)
        # print("YS ", ys)

        # print(">>>>> I " i)
        # print(matches)`

    def get_recall_accuracy(self, x, X, probeType):
        # TODO
        # get with 100 queries
        # matches_list = getquries()
        matches_list = lsh.get(X, collision_ratio=0.5, probeType=probeType)

        brute_forces = eval.bruteforce_get(x, X)
        avg_recall = 0
        count = 0

        for matches, answers in zip(matches_list, brute_forces):

            # print("MATCHES ", matches)

            # print("ANSWERS ", answers)

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
                features, inp_vec.iloc[idx].values.reshape(1, -1), metric='euclidean').flatten()

            nearest_neighbours = pd.DataFrame({'id': features.index, 'genre': tracks['track']['genre_top'].ix[features.index], 'distance': distance}).sort_values(
                'distance').reset_index(drop=True)

            # print("nearest negih")
            # print(nearest_neighbours.head())

            candidate_set_labels = nearest_neighbours.sort_values(
                by=['distance'], ascending=True)

            non_null = candidate_set_labels[candidate_set_labels['genre'].notnull(
            )]

            query_top_ks[idx] = non_null.iloc[:k]

        return query_top_ks

    def get_expected_genre_accuracy(self, all_data, inp_vec, probeType):

        matches_list = lsh.get(
            inp_vec, collision_ratio=0.5, probeType=probeType)

        ground_truths = tracks['track']['genre_top'].ix[inp_vec.index]

        ratio_sum = 0
        count = 0

        for answer, top_k_genres in zip(ground_truths, matches_list):

            ratio = self.get_answer_occurence(answer, top_k_genres)
            if not pd.isnull(answer):
                ratio_sum += ratio
                count += 1
                print("answer:", answer, ">> top:", top_k_genres)
            print("RATOIO ratio ", ratio)

        return ratio_sum / count, count

    def get_answer_occurence(self, answer, top_k_genres):

        if len(top_k_genres) == 0:
            return 0

        count = 0
        for genre in top_k_genres['genre']:
            if answer == genre:
                count += 1
        return count / len(top_k_genres)


X_train, X_test = train_test_split(features, test_size=10)
# get expected genre
# get number of correct in top 20
lsh = LSH.LSH(40, 25, 140)
# lsh.add(features['mfcc'])


# eval.eval_top_k_accuracy()
# query_df = features.iloc[1:2]
# brute_force_top_k = eval.bruteforce_get(
#     features['mfcc'], query_df['mfcc'])


# print("Brute-force : ", brute_force_top_k)


lsh.add(X_train['mfcc'])
eval = Evaluation(lsh)

# liszt = ft.compute_features("input_audio/ariana-grande.mp3")

# print("LSIZT", liszt)
# res_six = lsh.get(liszt['mfcc'], probeType="step-wise")


res = eval.get_recall_accuracy(
    X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")

# liszt = ft.compute_features("./input_audio/franz_list.mp3")
# res_six = lsh.get(liszt['mfcc'], probeType="step-wise")

# print(res_six)

# res, count = eval.get_expected_genre_accuracy(
#     X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")

# res = eval.get_boxplot_rand_projection(X_train['mfcc'])

# print("TOTAL accuracy ", res, " with no: ")

# val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# print("Process usage: ", val)
