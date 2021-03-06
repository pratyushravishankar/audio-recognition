# import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from collections import defaultdict
import scipy.io
import os
import requests
import numpy as np
import pandas as pd
import utils
# import IPython.display as ipd
import matplotlib
import resource


import features as ft
import heapq
from sklearn.metrics.pairwise import pairwise_distances

matplotlib.use('TkAgg')
fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))


# open file


# data = mat['X']
# data = Normalizer(norm='l2').fit_transform(data)
# data= data- data.mean(axis=0)
# classes = mat['X_class']

my_list = [1, 24, 21, 234, -8]

AUDIO_DIR = os.environ.get('data/fma_small')

tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')
features = utils.load('data/fma_metadata/features.csv')
echonest = utils.load('data/fma_metadata/echonest.csv')

# features = ''

n_vectors = 16


class LSH:

    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(
                HashTable(self.hash_size, self.inp_dimensions))

    def add(self, inp_vec, bitflip=False):
        for table in self.hash_tables:
            table.add(inp_vec, bitflip)

            val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # print("for each process Process usage: ", val)
        val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # print("AFTER FINISHING PROCESSES Process usage: ", val)

    def get(self, inp_vec, collision_ratio=0.1, probeType="rand_proj", k=0):

        # if probeType != "rand_proj":
        #     collision_ratio = 1

        # collisions_dict = {}
        queries_coll_list = [{} for i in range(len(inp_vec))]
        for table in self.hash_tables:

            table_matches = table.get(inp_vec, probeType, k)

            # matches for each vector in inp_vec
            for idx, query_match in enumerate(table_matches):

                # matches for specific query
                match_dict = queries_coll_list[idx]

                for m in query_match:
                    if m not in match_dict:
                        match_dict[m] = 0
                    match_dict[m] = match_dict[m] + 1

        query_matches = [[] for i in range(len(inp_vec))]

        for idx, query_dict in enumerate(queries_coll_list):
            for c in query_dict:
                if query_dict[c] >= self.num_tables * collision_ratio:
                    query_matches[idx].append(c)

        # return self.get_top_k(inp_vec, query_matches)
        return inp_vec, query_matches

    # TODO::// populate LSH databse with only non-nulls, so no don't need to filter top k by no-
    def get_top_k(self, inp_vec, candidates, k=20):

        if not candidates:
            return None

        # print("toplevel candidates len ", candidates)

        query_top_ks = [None for i in range(len(inp_vec))]

        for idx, cs in enumerate(candidates):
            ground_truths = tracks['track']['genre_top'].loc[cs]

            candidate_list = features.loc[cs]['mfcc']

            dists = []
            if len(candidate_list != 0):
                # distance = pairwise_distances(
                # candidate_list, inp_vec.iloc[idx], metric='euclidean').flatten()

                # for cand in candidate_list.values:
                #     distance = np.linalg.norm(cand, inp_vec.iloc[0])
                #     print("DISTANCE ", distance)

                # print("<ASDJLK", type(inp_vec.  [idx]))

                distances = pairwise_distances(
                    candidate_list, inp_vec.iloc[idx].values.reshape(1, -1))

                # print(">> ")

                for dist in distances:
                    dists.extend(dist)

            nearest_neighbours = pd.DataFrame({'id': cs, 'genre': ground_truths, 'distance': dists}).sort_values(
                'distance').reset_index(drop=True)

            candidate_set_labels = nearest_neighbours.sort_values(
                by=['distance'], ascending=True)

            non_null = candidate_set_labels[candidate_set_labels['genre'].notnull(
            )]

            top = min(k, len(non_null))
            query_top_ks[idx] = non_null[:top]

        return query_top_ks

        # candidate_list = features.ix[candidates]['mfcc']

        # # print("CANdidate list", candidate_list)
        # # print(candidate_list)

        # ground_truths = tracks['track']['genre_top'].ix[candidates]

        # distance = pairwise_distances(
        #     candidate_list, inp_vec, metric='euclidean').flatten()

        # nearest_neighbours = pd.DataFrame({'id': candidates, 'genre': ground_truths, 'distance': distance}).sort_values(
        #     'distance').reset_index(drop=True)

        # candidate_set_labels = nearest_neighbours.sort_values(
        #     by=['distance'], ascending=True)

        # non_null = candidate_set_labels[candidate_set_labels['genre'].notnull(
        # )]

        # top = min(k, len(non_null))
        # return non_null[:top]


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(inp_dimensions, hash_size)

    def add(self, inp_vec, bitflip=False):

        keys = self.get_exact_keys(inp_vec)

        if bitflip:
            keys = self.get_keys_cormode(inp_vec, k=2)
            # print("TOEHR KEY S+", other_keys)

        keys_df = keys.to_frame(name="idx")

        # print(keys_df)

        track_hashes = keys_df.join(tracks['track'])

        # TODO:// flip bits for dataset not working due to join

        # print(tracks.head())
        # print(tracks['track'].head())
        # track_hashes = keys_df.merge(
        #     tracks['track'], left_on='idx', right_index=True)
        # # print(track_hashes)

        for track in track_hashes.itertuples():

            # hash
            key = track[1]

            # song_id
            val = track[0]
            if key not in self.hash_table:
                self.hash_table[key] = []
            self.hash_table[key].append(val)

    def get_keys(self, inp_vec, is_probe):

        bin_bits = inp_vec.dot(self.projections) >= 0

        # TODO make a dict for each input vecotr

        # print("bits ", bin_bits)

        # print("orig_key ", orig_keys)

        if (is_probe):
            # print("BEFORE")
            # print(len(bin_bits))
            bin_bits = self.get_probe_bins(bin_bits)
            # print("AFTER")
            # print(len(bin_bits))
            # print(bin_bits)

        powers_of_two = 1 << np.arange(self.hash_size - 1, -1, step=-1)
        decimal_keys = bin_bits.dot(powers_of_two)
        return decimal_keys

    def get_exact_keys(self, inp_vec):

        bin_bits = inp_vec.dot(self.projections) >= 0

        # TODO make a dict for each input vecotr

        # print("bits ", bin_bits)

        # print("orig_key ", orig_keys)

        powers_of_two = 1 << np.arange(self.hash_size - 1, -1, step=-1)
        decimal_keys = bin_bits.dot(powers_of_two)
        return decimal_keys

    def get_keys_cormode(self, inp_vec, k=0):

        # print("AT CORMODE!!!!!")
        bin_bits = inp_vec.dot(self.projections)

        # print("bin bits before ", bin_bits)
        probed_keys = []
        ixs = []

        # print(" BIN_BITS", bin_bits)

        for i, row in bin_bits.iterrows():

            # print("R OW ", row)

            abs_idxs = [(abs(val), idx) for idx, val in enumerate(row)]

            smallest = heapq.nsmallest(k, abs_idxs)
            probed_keys.append(row)
            ixs.append(i)
            for pair in smallest:
                idx = pair[1]
                row_copy = row.copy()
                row_copy[idx] = row_copy[idx] * -1
                probed_keys.append(row_copy)
                ixs.append(i)

        probed_projections = pd.DataFrame(
            np.array(probed_keys), index=ixs, columns=bin_bits.columns)

        # print("probed_projections", probed_projections)
        probed_bin_bits = (probed_projections) >= 0

        powers_of_two = 1 << np.arange(self.hash_size - 1, -1, step=-1)
        decimal_keys = probed_bin_bits.dot(powers_of_two)
        return decimal_keys


# return key

        # print(bin_bits)
#


    def get_probe_bins(self, bin_indices_bits, search_radius=1):

        bin_bits = bin_indices_bits.dot(self.projections) >= 0

        # print("S> >", bin_bits.shape)

        p_bins = []
        for orig_bin in bin_bits.values:

            p_bins.append(orig_bin)

            for perturbed_idxs in combinations(range(self.hash_size), search_radius):

                idxs = list(perturbed_idxs)
                # print("pertubrs ", idxs)
                perturbed_query = orig_bin.copy()

                for idx in idxs:
                    perturbed_query[idx] = not perturbed_query[idx]

                p_bins.append(perturbed_query)

        probed_bin_bits = pd.DataFrame(
            np.array(p_bins), columns=bin_bits.columns)
        powers_of_two = 1 << np.arange(self.hash_size - 1, -1, step=-1)
        decimal_keys = probed_bin_bits.dot(powers_of_two)
        return decimal_keys

    def get(self, inp_vec, probeType, k=0):

        res = []
        if probeType == "step-wise":
            # print("step -wise!!!")
            # bins = self.get_keys(inp_vec, True)
            bins = self.get_probe_bins(inp_vec, search_radius=1)

            # change to permutation
            len_same_query_bins = self.hash_size + 1

            for i in range(0, len(bins), len_same_query_bins):

                same_queries_matches = []
                for j in range(len_same_query_bins):

                    idx = i + j
                    key = bins[idx]
                    if key in self.hash_table:
                        same_queries_matches.extend(self.hash_table[key])

                res.append(same_queries_matches)
                # TODO get contentes form each bin

        elif probeType == "bit-flip":
            # print("bit-flip!!!")

            bins = self.get_keys_cormode(inp_vec, k=k)

            # print("bit flip bins ", bins)

            len_same_query_bins = k + 1

            for i in range(0, len(bins), len_same_query_bins):

                same_queries_matches = []
                for j in range(len_same_query_bins):

                    idx = i + j

                    # print("i + J", idx)
                    key = bins.iloc[idx]
                    if key in self.hash_table:
                        same_queries_matches.extend(self.hash_table[key])

                res.append(same_queries_matches)
            # print("RES bitflip: ", res)

            # print("BINS >>> ", bins)
        else:
            # bins = self.get_keys(inp_vec, False)
            bins = self.get_exact_keys(inp_vec)

            # print("bins: ", bins)

            for key in bins:

                if key in self.hash_table:
                    # return self.hash_table[key]
                    res.append(self.hash_table[key])

        return res

        # print("BINS", bins)

        # print("BINS bins, bins ", bins)

    def majority(self, bin):

        # print("BIN", bin)

        genre_count = {}

        max_val = ""
        max_count = 0
        for g in bin:
            if g not in genre_count:
                genre_count[g] = 0
            genre_count[g] = genre_count[g] + 1

            # if (genre_count[g] > max_count):
            #     max_val = g
            #     max_count = genre_count[g]
        return genre_count

    def bar_chart(self, key, genres):

        genre_count = {}
        count = 0
        for g in genres:
            count = count + 1
            if g not in genre_count:
                genre_count[g] = 0
            genre_count[g] = genre_count[g] + 1

        max_val = ""
        val = 0
        for g in genre_count:

            if genre_count[g] > val:
                max_val = g
                val = genre_count[g]
        if (max_val == "Electronic"):
            print("key ", key, " values : ", genre_count,
                  " total: ", count, " \n \n \n")


lsh = LSH(1, 25, 140)
lsh.add(features['mfcc'])

# lsh_two = LSH(40, 25, 140)
# # lsh_two.add(features['mfcc'])


# X_train, X_test = train_test_split(features, test_size=100)

# print("SHAPES ", X_train.shape)
# print(X_test.shape)
# # query_df = ft.compute_features(2)


# kate_nash_10s = ft.compute_features("./input_audio/kate_nash_10s.mp3")
# # kate_nash_5s = ft.compute_features("./input_audio/kate_nash_5s.mp3")
# # tinie = ft.compute_features("./input_audio/tinie.mp3")
# # kate_nash_full = ft.compute_features("./input_audio/kate_nash_full.mp3")
# # ariana_grande = ft.compute_features("./input_audio/ariana-grande.mp3")
# # ludovico = ft.compute_features("./input_audio/ludovico_einaudi.mp3")
# # liszt = ft.compute_features("./input_audio/franz_list.mp3")
# # second = ft.compute_features("./input_audio/test.mp3")

# print("KATE/ NASHSES")
# print(kate_nash_10s)
# # print("5sssss")
# # print(kate_nash_5s)
# # print("kate_nash_full")
# # print(kate_nash_full)
# # print("tinie")
# # print(tinie)

# # print("MY FEATURES")
# # print(query_df)
# # print("THEIRS")
# print(features.iloc[1:2])


# res = lsh.get(kate_nash_10s['mfcc'])
# print("KATE_NASH 10s")
# print(res)

# # res_two = lsh.get(kate_nash_5s)
# # print("KATE_NASH 5s")
# # print(res_two)

# # res_three = lsh.get(kate_nash_full)
# # print("KATE_NASH FULL ")
# # print(res_three)


# # res_four = lsh.get(ariana_grande)
# # print("ARIANA ")
# # print(res_four)

# # res_five = lsh.get(ludovico)
# # print("LUDOVICO")
# # print(res_five)


# dummy = lsh.get(features.iloc[1:3]['mfcc'], probeType=False)
# print("DUMMY - second song")
# for d in dummy:
#     print(">> ", d)


# # liszt = ft.compute_features("./input_audio/franz_list.mp3")
# # res_six = lsh.get(liszt['mfcc'], probeType="step-wise")
# # print("step-wise")
# # print(res_six)


# # liszttwo = ft.compute_features("./input_audio/franz_list.mp3")
# # cormode = lsh.get(liszttwo['mfcc'], probeType="bit-flip")
# # print(">> bit-flip")
# # print(cormode)


# liszttwo = ft.compute_features("./input_audio/franz_list.mp3")


# rand_proj = lsh.get(liszttwo['mfcc'], probeType="rand-proj")
# print("rand_projj one table")
# print(rand_proj)

# rand_proj_two = lsh_two.get(liszttwo['mfcc'], probeType="rand-proj")
# print("rand_projj multiple!!!!! table")
# print(rand_proj_two)

# # res_eight = lsh.get(second)
# # print("my features - second song")
# # print(res_eight)
