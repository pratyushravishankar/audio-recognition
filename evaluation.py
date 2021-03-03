
import lsh_random_projection as LSH
import time
import features as ft
import utils
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

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
        # print("RAND PROJ ", lsh_random_proj_top_k)
        # print("STEP-wise ", lsh_probe_step_wise_top_k)
        # print("bit flip ", lsh_probe_bit_flip_top_k)

        # spectral_top_k_score =

    def get_search_quality(self, ys, Ys):

        k = len(ys)

        print("STRAT")

        count = 0
        for Y in Ys:
            if (ys == Y).any():

                print("FOUND ", Y)
                count = count + 1

        return count / k

    def bruteforce_get(self, features, inp_vec, k=20):

        distance = pairwise_distances(
            features, inp_vec, metric='euclidean').flatten()

        nearest_neighbours = pd.DataFrame({'id': features.index, 'genre': tracks['track']['genre_top'], 'distance': distance}).sort_values(
            'distance').reset_index(drop=True)

        # print("nearest negih")
        # print(nearest_neighbours.head())

        candidate_set_labels = nearest_neighbours.sort_values(
            by=['distance'], ascending=True)

        non_null = candidate_set_labels[candidate_set_labels['genre'].notnull(
        )]

        return non_null.iloc[:k]


lsh = LSH.LSH(1, 25, 140)
lsh.add(features['mfcc'])
eval = Evaluation(lsh)

eval.eval_top_k_accuracy()
