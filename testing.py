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
import heapq
from sklearn.metrics.pairwise import pairwise_distances

matplotlib.use('TkAgg')
fma = utils.FreeMusicArchive(os.environ.get('FMA_KEY'))


AUDIO_DIR = os.environ.get('data/fma_small')

tracks = utils.load('data/fma_metadata/tracks.csv')
genres = utils.load('data/fma_metadata/genres.csv')
features = utils.load('data/fma_metadata/features.csv')
echonest = utils.load('data/fma_metadata/echonest.csv')


def test(features, inp_vec, k=20):

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


# second = ft.compute_features("./input_audio/test.mp3")
# res_eight = test(features['spectral_centroid', 'zcr', 'rmse'], second['spectral_centroid', 'zcr', 'rmse'])
# print("My features  - second song")
# print(res_eight)


# kate_nash_10s = ft.compute_features("./input_audio/kate_nash_10s.mp3")
# kate_nash_5s = ft.compute_features("./input_audio/kate_nash_5s.mp3")
# tinie = ft.compute_features("./input_audio/tinie.mp3")
# kate_nash_full = ft.compute_features("./input_audio/kate_nash_full.mp3")
# ariana_grande = ft.compute_features("./input_audio/ariana-grande.mp3")
# ludovico = ft.compute_features("./input_audio/ludovico_einaudi.mp3")
# liszt = ft.compute_features("./input_audio/franz_list.mp3")
# second = ft.compute_features("./input_audio/test.mp3")
dummy = features.iloc[1289:1290]


# res = test(features['mfcc'],
#            kate_nash_10s['mfcc'])
# print("KATE_NASH 10s")
# print(res)

# res_two = test(features['mfcc'],
#                kate_nash_5s['mfcc'])
# print("KATE_NASH 5s")
# print(res_two)

# res_three = test(features['mfcc'],
#                  kate_nash_full['mfcc'])
# print("KATE_NASH FULL ")
# print(res_three)


# res_four = test(features['mfcc'],
#                 ariana_grande['mfcc'])
# print("ARIANA ")
# print(res_four)

# res_five = test(features['mfcc'],
#                 ludovico['mfcc'])
# print("LUDOVICO")
# print(res_five)

# res_six = test(features['mfcc'],
#                liszt['mfcc'])
# print("LISZT")
# print(res_six)


res_seven = test(features['mfcc'], dummy['mfcc'])
print("DUMMY - second song")
print(res_seven)

# res_eight = test(features['mfcc'], second['mfcc'])
# print("my features - second song")
# print(res_eight)
