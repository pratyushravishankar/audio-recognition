# import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from collections import defaultdict
import scipy.io
import os
import requests
import numpy as np
import pandas as pd
import utils
import IPython.display as ipd
import matplotlib
import features as ft

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

    def add(self, inp_vec):
        for table in self.hash_tables:
            table.add(inp_vec)

    def get(self, inp_vec):
        # insert further logic
        res = []
        for table in self.hash_tables:
            res.append(table.get(inp_vec))
        return res


class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(inp_dimensions, hash_size)

    def add(self, inp_vec):
        # bin_indices_bits = inp_vec.dot(self.projections) >= 0
        keys = self.get_keys(inp_vec)
        # print("KEY", keys)

        track_hashes = keys.join(tracks['track'])
        for track in track_hashes.itertuples():
            # print("track ", track)
            key = track[1]
            val = track[9]

            # print("key", key, " val: ", val)

            if key not in self.hash_table:
                self.hash_table[key] = []
            self.hash_table[key].append(val)

        for i in self.hash_table:
            vals = self.hash_table[i]
            self.bar_chart(i, vals)

    def get_keys(self, inp_vec):
        # print("INPUT ", inp_vec)
        bin_indices_bits = inp_vec.dot(self.projections) >= 0
        powers_of_two = 1 << np.arange(self.hash_size - 1, -1, step=-1)
        bin_indices = bin_indices_bits.dot(powers_of_two)
        # print(bin_indices)
        # return str(bin_indices)
        return bin_indices.to_frame(name="idx")

    def get(self, inp_vec):

        res = []
        bins = self.get_keys(inp_vec)
        print("TINIES keyeys")
        print(bins)
        for binquery in bins.itertuples():
            print("ACC BIN: ", binquery)
            query_bin = binquery[1]
            if query_bin in self.hash_table:
                res.append(self.hash_table[query_bin])
                print("ACC MAJORITY")
                print(self.majority(self.hash_table[query_bin]))
            else:
                res.append("NULL")

        return res

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


lsh = LSH(1, 25, 518)
lsh.add(features)

query_df = ft.compute_features(2)

res = lsh.get(query_df)
# print("THE RESPONSE")
# print(res)
