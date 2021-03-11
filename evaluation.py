
import matplotlib
from tqdm import tqdm
import librosa
from scipy import stats
import warnings
import multiprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import utils
import features as ft
import time
import lsh_random_projection as LSH
import resource
import numpy as np
import seaborn as sns
import os
import sklearn.preprocessing
import librosa.display
# import matplotlib.pyplot as

features = utils.load('data/fma_metadata/features.csv')
tracks = utils.load('data/fma_metadata/tracks.csv')


class Evaluation:

    # pass in lsh table instances for spectral, probe, ...
    def __init__(self, lsh):
        self.lsh = lsh

    # def get_accuracy

    # lsh_probe_1
    # lsh_probe_2
    # spectral/
    # lsh_ranodmised_proj

    # def get_list_times(querymethod,  queries):
    #     None

    # for q in queries:
    # tic = time.perf_counter()        # lsh.get(q)
    # toc = time.perf_counter()
    # time.list.append(toc - tic)

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

    # np.random.seed(19680801)

    # # fake up some data
    # spread = np.random.rand(50) * 100
    # center = np.ones(25) * 50
    # flier_high = np.random.rand(10) * 100 + 100
    # flier_low = np.random.rand(10) * -100
    # data = np.concatenate((spread, center, flier_high, flier_low))
    # fig1, ax1 = plt.subplots()
    # ax1.set_title('Basic Plot')
    # ax1.boxplot(data)
    # plt.show()

    #
    # print("XS ", xs)
    # print("YS ", ys)

    # print(">>>>> I " i)
    # print(matches)`

    def get_recall_accuracy(self, x, X, probeType, k=0):
        # TODO
        # get with 100 queries
        # matches_list = getquries()
        matches_list = self.lsh.get(
            X, collision_ratio=0.5, probeType=probeType, k=k)

        brute_forces = self.bruteforce_get(x, X)
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


# def get_accuraccy_over_hashtables


def pca():

    X_train, X_test = train_test_split(
        features['mfcc'], test_size=0.2, random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    plt.scatter(X_test[:, 0], X_test[:, 1], edgecolor='none', alpha=0.5,
                )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()

    plt.show()


def grid_search():

    key_sizes = [i for i in range(5, 40, 5)]
    tables_sizes = [i for i in range(1, 50, 5)]

    X_train, X_test = train_test_split(features, test_size=10)

    res = []
    res_keys = []
    res_tables = []

    acc = []

    for key in key_sizes:
        for table in tables_sizes:

            lsh = LSH.LSH(table, key, 140)
            lsh.add(X_train['mfcc'])

            eval = Evaluation(lsh)

            accuracy = eval.get_recall_accuracy(
                X_train['mfcc'], X_test['mfcc'], "rand_proj")

            res.append([key, table, accuracy])

            acc.append(accuracy)
            res_keys.append(key)
            res_tables.append(table)

        # acc.append("\n")

    print(acc)

    data = pd.DataFrame(
        data={'keys': res_keys, 'tables': res_tables, 'accuracy': acc})
    data = data.pivot(index='keys', columns='tables', values='accuracy')

    sns.set_style("whitegrid")
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 1, 1)
    sns.heatmap(data, annot=True, cmap="YlGnBu").set_title(
        'Random Projection LSH grid search')
    plt.show()

    # mpl.rcParams['figure.figsize'] = (8.0, 7.0)
    # sns.heatmap(grid_search_groupby(results, 'max_depth', 'n_estimators'),
    #             cmap='plasma', annot=True, fmt='.4f')
    # plt.title('Grid Search Result: Max Depth vs N-Estimators')
    # plt.xlabel('N_Estimators')
    # plt.ylabel('Max Depth')
    # plt.figure(figsize=(8, 6))
    # plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    # plt.imshow(acc, interpolation='nearest', cmap=plt.cm.hot)
    # plt.xlabel('n_estimators')
    # plt.ylabel('min_samples_leaf')
    # plt.colorbar()
    # plt.xticks(np.arange(len(n_estimators)), n_estimators)
    # plt.yticks(np.arange(len(min_samples_leaf)), min_samples_leaf)
    # plt.title('Grid Search AUC Score')
    # plt.show()


# grid_search()


X_train, X_test = train_test_split(features, test_size=10)
# # get expected genre
# # get number of correct in top 20

# val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# print("BEFORE Process usage: ", val)

# lsh = LSH.LSH(1, 16, 140)
# lsh.add(features['mfcc'])


# # eval.eval_top_k_accuracy()
# # query_df = features.iloc[1:2]
# # brute_force_top_k = eval.bruteforce_get(
# #     features['mfcc'], query_df['mfcc'])


# # print("Brute-force : ", brute_force_top_k)


# lsh.add(X_test['mfcc'], bitflip=True)
# eval = Evaluation(lsh)

# liszt = ft.compute_features("output.wav")

# print("LSIZT", liszt['mfcc'])
# res_six = lsh.get(liszt['mfcc'], probeType="step-wise")


# res = eval.get_recall_accuracy(
# X_train['mfcc'], X_test['mfcc'], probeType="bit-flip")

# liszt = ft.compute_features("./input_audio/franz_list.mp3")
# res_six = lsh.get(liszt['mfcc'], probeType="step-wise")


# print(res)

# print(liszt)

# res, count = eval.get_expected_genre_accuracy(
#     X_train['mfcc'], X_test['mfcc'], probeType="bit-flip")

# # res = eval.get_boxplot_rand_projection(X_train['mfcc'])

# print("TOTAL accuracy ", res, " with no: ")

# # val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# # print("Process usage: ", val)

# val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# print("AFTER!!!! Process usage: ", val)


# plt.rcParams['figure.figsize'] = (18, 4)

# x, fs = librosa.load("output.wav")
# librosa.display.waveplot(x, sr=fs)

# mfccs = librosa.feature.mfcc(x, sr=fs)
# print(mfccs.shape)
# print(mfccs)
# mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
# plt.show()

def recall_probes():

    key_size = 16

    lsh_step_wise = LSH.LSH(10, key_size, 140)
    lsh_step_wise.add(X_train['mfcc'])

    lsh_bit_flip = LSH.LSH(10, key_size, 140)
    lsh_bit_flip.add(X_train['mfcc'], True)

    eval_step_wise = Evaluation(lsh_step_wise)
    eval_bit_flip = Evaluation(lsh_bit_flip)

    step_wise_res = eval_step_wise.get_recall_accuracy(
        X_train['mfcc'], X_test['mfcc'], probeType="step-wise")
    step_wise_no_probes = [key_size]

    bit_flip_res = []
    bit_flip_no_probes = []
    for i in range(5):
        print("i:", i)
        res = eval_bit_flip.get_recall_accuracy(
            X_train['mfcc'], X_test['mfcc'], probeType="bit-flip", k=i+1)
        bit_flip_res.append(res)
        bit_flip_no_probes.append(i + 1)

    print("step-wise res", step_wise_res)
    print("bit-flip res", bit_flip_res)

    plt.plot(bit_flip_res, bit_flip_no_probes,
             color='red', marker='o', label="bit-flip")
    plt.plot(step_wise_res, step_wise_no_probes,
             color='blue', marker='x', label="step-wise")
    plt.title('Avg recall for each number of probes ', fontsize=14)
    plt.xlabel('No. of probes', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def pca():

    X_train, X_test = train_test_split(
        features['mfcc'], test_size=0.05, random_state=0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    plt.scatter(X_test[:, 0], X_test[:, 1], edgecolor='none', alpha=0.5,
                )
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.title("PCA of FMA audio dataset")
    plt.colorbar()

    plt.show()


# def get accuracy():
recall_probes()
