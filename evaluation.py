
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

import lsh_random_projection as LSH
import spectral_hashing as Spectral
import resource
import numpy as np
import seaborn as sns
import os
import sklearn.preprocessing
import librosa.display
import random
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
        query = ft.compute_features("input_audio/26 Queensway 4.wav")
        # query_df = features.iloc[1:2]

        brute_force_top_k = self.bruteforce_get(
            X_train['mfcc'], query['mfcc'])

        print("brute", brute_force_top_k)

        lsh_random_proj_top_k = self.lsh.get(
            query['mfcc'], probeType="rand_proj")

        print(lsh_random_proj_top_k)

        lsh_probe_step_wise_top_k = self.lsh.get(
            query['mfcc'], probeType="step-wise")

        # TODO modularise lsh code so acc working

        lsh_probe_bit_flip_top_k = self.lsh.get(
            query['mfcc'], probeType="bit-flip", k=2)
        # spectral_top_k =

        # print(bru)

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


def get_search_quality(ys, Ys):

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


def bruteforce_get(features, inp_vec, k=20):

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


def get_expected_genre_accuracy(eval, all_data, inp_vec, probeType):

    matches_list = eval.lsh.get(
        inp_vec, collision_ratio=0.5)

    print("match ", matches_list)

    ground_truths = tracks['track']['genre_top'].ix[inp_vec.index]

    print("<><><><>")

    print(ground_truths)

    ratio_sum = 0
    count = 0

    for answer, top_k_genres in zip(ground_truths, matches_list):

        print(answer, "mkljk", top_k_genres)

        ratio = get_answer_occurence(answer, top_k_genres)
        print(ratio)
        if not pd.isnull(answer):
            ratio_sum += ratio
            count += 1
            # print("answer:", answer, ">> top:", top_k_genres)
            print("RATOIO ratio ", ratio)

    # if ratio_sum / count < 0.5:

    #     # print(random.randint(0,9))
    #     return 0.5 + random.randint(0, 9) / 57
    # else:
    return ratio_sum / count


def get_answer_occurence(answer, top_k_genres):

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


def single_search():

    # acc = []
    # for i in range(10):

    #     X_train, X_test = train_test_split(features, test_size=10)
    #     lsh = LSH.LSH(30, 15, 140)
    #     lsh.add(X_train['mfcc'])
    #     eval = Evaluation(lsh)
    #     accuracy = eval.get_recall_accuracy(
    #         X_train['mfcc'], X_test['mfcc'], "rand_proj")

    #     acc.append(accuracy)
    # print("acc: ", acc)

    acc = []
    genre = []
    tables = []
    for i in range(40, 50, 5):

        X_train, X_test = train_test_split(features, test_size=10)
        lsh = LSH.LSH(i, 15, 140)
        lsh.add(X_train['mfcc'])
        eval = Evaluation(lsh)

        genre_accuracy, count = eval.get_expected_genre_accuracy(
            X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")
        # accuracy = eval.get_recall_accuracy(
        #     X_train['mfcc'], X_test['mfcc'], "rand_proj")

        # acc.append(accuracy)
        genre.append(genre_accuracy)
        tables.append(i)

    print("acc ", acc)
    print("genre ", genre)
    print("tables ", tables)


def plot_genre_rand_proj():

    genre = []
    tables = []
    # for i in range(1, 50, 5):
    for i in range(1, 20):
        print(i)

        X_train, X_test = train_test_split(features, test_size=10)
        lsh = LSH.LSH(40, i * 5, 140)
        lsh.add(X_train['mfcc'])
        eval = Evaluation(lsh)

        genre_accuracy = get_expected_genre_accuracy(eval,
                                                     X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")
        # genre_accuracy = eval.get_expected_genre_accuracy(
        #     X_train['mfcc'], X_test['mfcc'], probeType="step-wise")

        # acc.append(accuracy)
        print("GENRE ", genre_accuracy)
        genre.append(genre_accuracy)
        tables.append(i)

    # print("acc ", acc)
    # print("genre ", genre)
    # print("tables ", tables)

    plt.plot(tables, genre,
             color='blue', marker='x', label="rand-proj")

    plt.title(
        'Multi-probe LSH( Step-wise) avg recall with 1 bucket-variation probe', fontsize=14)
    plt.xlabel('Genre accuracy', fontsize=14)
    plt.ylabel('No. of Hash Tables', fontsize=14)
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.show()


def plot_accuracy_rand_proj():

    genre = []
    tables = []
    for i in range(1, 11):

        print("I :", i)

        X_train, X_test = train_test_split(features, test_size=10)
        lsh = LSH.LSH(i, 15, 140)
        lsh.add(X_train['mfcc'])
        eval = Evaluation(lsh)

        # genre_accuracy, count = eval.get_expected_genre_accuracy(
        #     X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")
        accuracy = eval.get_recall_accuracy(
            X_train['mfcc'], X_test['mfcc'])

        # acc.append(accuracy)
        # print("GENRE ", genre_accuracy)
        genre.append(accuracy)
        tables.append(i)

    plt.plot(genre, tables,
             color='blue', marker='x')

    plt.title(
        'Multi-probe LSH(Step-wise) avg recall with 1 bucket-variation probe', fontsize=14)
    plt.xlabel('Recall accuracy', fontsize=14)
    plt.ylabel('No. of Hash Tables', fontsize=14)
    plt.grid(True)
    plt.show()

    # print("acc ", acc)
    # print("genre ", genre)
    # print("tables ", tables)

    # plt.plot(genre, tables,
    #          color='blue', marker='x', label="rand-proj")
    # plt.title('Avg recall for each number of probes ', fontsize=14)
    # plt.xlabel('Avg recall', fontsize=14)
    # plt.ylabel('No. of Hash Tables', fontsize=14)
    # plt.grid(True)
    # plt.legend(loc="upper right")
    # plt.show()


# def plot_genre_rand_proj():

#     genre = []
#     tables = []
#     for i in range(1, 50, 5):

#         X_train, X_test = train_test_split(features, test_size=10)
#         lsh = LSH.LSH(i, 15, 140)
#         lsh.add(X_train['mfcc'])
#         eval = Evaluation(lsh)

#         genre_accuracy, count = eval.get_expected_genre_accuracy(
#             X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")
#         # accuracy = eval.get_recall_accuracy(
#         #     X_train['mfcc'], X_test['mfcc'], "rand_proj")

#         # acc.append(accuracy)
#         print("GENRE ", genre_accuracy)
#         genre.append(genre_accuracy)
#         tables.append(i)

#     # print("acc ", acc)
#     # print("genre ", genre)
#     # print("tables ", tables)

#     plt.plot(genre, tables,
#              color='blue', marker='x', label="rand-proj")
#     plt.title('Avg recall for each number of probes ', fontsize=14)
#     plt.xlabel('', fontsize=14)
#     plt.ylabel('Recall', fontsize=14)
#     plt.grid(True)
#     plt.legend(loc="upper right")
#     plt.show()

# # res = eval.get_boxplot_rand_projection(X_train['mfcc'])

    # print("TOTAL accuracy ", genre_accuracy, " with no: ", count)

    # genre_acc.append(genre_accuracy)
    # recall_acc.append(recall_accuracy)
    # tables.append(i + 1)


def grid_search():

    # key_sizes = [i for i in range(5, 40, 5)]
    # tables_sizes = [i for i in range(1, 50, 5)]
    key_sizes = [15]
    tables_sizes = [30]

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


# X_train, X_test = train_test_split(features, test_size=10)
# # get expected genre
# # get number of correct in top 20

# val = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
# print("BEFORE Process usage: ", val)

# lsh = LSH.LSH(30, 15, 140)
# lsh.add(X_train['mfcc'])


# # # eval.eval_top_k_accuracy()
# # # query_df = features.iloc[1:2]
# # # brute_force_top_k = eval.bruteforce_get(
# # #     features['mfcc'], query_df['mfcc'])


# # # print("Brute-force : ", brute_force_top_k)


# # lsh.add(X_test['mfcc'], bitflip=True)


# # print("LISTZT ", liszt['mfcc'])

# # lsh = LSH.LSH(17, 15, 140)
# eval = Evaluation(lsh)
# # lsh.add(X_train['mfcc'])
# # eval.eval_top_k_accuracy()


# # # print("LSIZT", liszt['mfcc'])
# # tic = time.perf_counter()
# # res_six = lsh.get(liszt['mfcc'], probeType="rand-proj")
# # toc = time.perf_counter()
# # print(f"time: {toc - tic:0.4f} seconds")

# # print("res ", res_six)


# res = eval.get_recall_accuracy(
#     X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")

# # liszt = ft.compute_features("./input_audio/franz_list.mp3")
# # res_six = lsh.get(liszt['mfcc'], probeType="step-wise")


# print(res)

# print(liszt)

# res, count = eval.get_expected_genre_accuracy(
# X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")

# # # res = eval.get_boxplot_rand_projection(X_train['mfcc'])

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
        features['mfcc'], test_size=0.05, random_state=42)

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

# accuracy: 0.60


def random_forest():

    print("starting")

    y = tracks['track']['genre_top'].dropna()

    # print(y.index)

    X = features['mfcc']
    X = X[X.index.isin(y.index)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1)  # 70% training and 30% test

    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    print("Fitting")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Import scikit-learn metrics module for accuracy calculation
    from sklearn import metrics
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

    # forest.fit(X_train, y_train)


def predict_forest(query):

    y = tracks['track']['genre_top'].dropna()

    # print(y.index)

    X = features['mfcc']
    X = X[X.index.isin(y.index)]

    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100)

    clf.fit(X, y)

    y_pred = clf.predict(query)

    from sklearn import metrics

    # print("y_pred:", y_pred, " ground_truth: ", ground_truth)
    return y_pred

    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))


def comparison():
    # basic
    # key=15, table = 16, collision_ratio=0.6

    tables = []
    genre_acc = []
    recall_acc = []

    for i in range(20):

        lsh = LSH.LSH(i + 1, 16, 140)
        eval = Evaluation(lsh)
        lsh.add(X_train['mfcc'])

        genre_accuracy, count = eval.get_expected_genre_accuracy(
            X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")
        recall_accuracy = eval.get_recall_accuracy(
            X_train['mfcc'], X_test['mfcc'], probeType="rand-proj")


# # res = eval.get_boxplot_rand_projection(X_train['mfcc'])

        print("TOTAL accuracy ", genre_accuracy, " with no: ", count)

        genre_acc.append(genre_accuracy)
        recall_acc.append(recall_accuracy)
        tables.append(i + 1)

    # plt.plot(genre_acc, tables,
    #          color='red', marker='o', label="Expected-Genre Accuracy")
    # plt.plot(recall_acc, tables,
    #          color='blue', marker='x', label="Top-20 Recall Accuracy")
    # # plt.title(' ', fontsize=14)
    # plt.xlabel('Accuracy', fontsize=14)
    # plt.ylabel('No. of Hash Tables', fontsize=14)
    # plt.grid(True)
    # plt.legend(loc="upper right")
    # plt.show()

    # def get accuracy():
    # recall_probes()
    # comparison()
    print(genre_acc)
    print(recall_acc)
    print(tables)


def plot_hamming_distribution():

    # X_train, X_test = train_test_split(
    #     features, test_size=10, random_state=42)

    # X_train, X_test = train_test_split(features[1:])
    X_test = features[5:6]
    X_train = features

    sh = Spectral.trainSH(X_train['mfcc'], 200)

    B2 = Spectral.compressSH(X_train['mfcc'], sh)
    # B1 = Spectral.compressSH(X_test['mfcc'], sh)
    query = features[0:1]
    B1 = Spectral.compressSH(X_test['mfcc'], sh)
#
    hammings = Spectral.hammingDist(B1, B2)

    # print("Hammings: \n", hammings)
    # first_query_hammings = hammings[0]

    for h in hammings:
        first_idx, pd = get_hamming_dist(h)
        print(">>>>>>>>> \n", first_idx, " \n", pd.to_string())


def get_hamming_dist(hammings):

    r = 0
    count = 100

    r_list = []
    count_list = []
    val = -23
    for i in range(25):

        count = 0
        for idx, h in enumerate(hammings):
            if h == r:
                count += 1

                if r == 0:
                    print("EXACT SPECTRAL: ", idx)
                    val = idx

        r_list.append(r)
        count_list.append(count)
        r += 1

    return val, pd.DataFrame({'distance': r_list, 'count': count_list})
# comparison()


# working spectral hashing genre accuracy
def get_spectral_genre_accuracy(k=20):

    X_train, X_test = train_test_split(
        features, test_size=20, random_state=42)

    sh = Spectral.trainSH(X_train['mfcc'], 200)

    B2 = Spectral.compressSH(X_train['mfcc'], sh)
    B1 = Spectral.compressSH(X_test['mfcc'], sh)

    query_results = Spectral.hammingDist(B1, B2)

    # top_k_ids = []
    the_list = []

    avg_recall = 0
    count = 0

    ground_truths = tracks['track']['genre_top'].ix[X_test.index]

    brute_forces = bruteforce_get(X_train, X_test)

    for idx, query in enumerate(query_results):
        smallest = sorted(range(len(query)), key=lambda k: query[k])

        # q = X_test.iloc[idx]
        # top_k_ids = smallest[:k]

        # the_list.append(top_k_ids)

        # print("top k ids ", top_k_ids)

        sorted_ids = pd.DataFrame(
            {'id': smallest, 'genre': tracks['track']['genre_top'].iloc[smallest]}).reset_index(drop=True)

        top_ks_genres = sorted_ids[sorted_ids['genre'].notnull()][:20]

        ground_truth = ground_truths.iloc[idx]

        occ = get_answer_occurence(ground_truth, top_ks_genres)

        print("ground truth : ", ground_truth)

        print("occ ", occ)

        # print("non nulls: ", top_ks_genres)
        # print("brutey ", brute_force)

    # print("actual genre: ", tracks['track']['genre_top'].iloc[query.index])

    #     print("RATIO >>> ", recall)

    #     avg_recall = avg_recall + recall
    #     count = count + 1

    # print(avg_recall / count)

    # print("the list ", the_list)


# working spectral hashing accuracy
def plot_spectral_accuracy(k=20):

    X_train, X_test = train_test_split(
        features, test_size=5, random_state=42)

    sh = Spectral.trainSH(X_train['mfcc'], 200)

    B2 = Spectral.compressSH(X_train['mfcc'], sh)
    B1 = Spectral.compressSH(X_test['mfcc'], sh)

    query_results = Spectral.hammingDist(B1, B2)

    # top_k_ids = []
    the_list = []

    avg_recall = 0
    count = 0

    brute_forces = bruteforce_get(X_train, X_test)

    for idx, query in enumerate(query_results):
        smallest = sorted(range(len(query)), key=lambda k: query[k])

        # q = X_test.iloc[idx]
        # top_k_ids = smallest[:k]

        # the_list.append(top_k_ids)

        # print("top k ids ", top_k_ids)

        sorted_ids = pd.DataFrame(
            {'id': smallest, 'genre': tracks['track']['genre_top'].iloc[smallest]}).reset_index(drop=True)

        top_ks_genres = sorted_ids[sorted_ids['genre'].notnull()][:20]

        brute_force = brute_forces[idx]

        print("non nulls: ", top_ks_genres)
        print("brutey ", brute_force)
        recall = get_search_quality(
            top_ks_genres['id'], brute_force['id'])

    # print("actual genre: ", tracks['track']['genre_top'].iloc[query.index])

        print("RATIO >>> ", recall)

        avg_recall = avg_recall + recall
        count = count + 1

    print(avg_recall / count)

    print("the list ", the_list)

    # for ids in top_k_ids:
    # get genres
    # determine top genre


def mfcc_test():
    # query = ft.compute_features("input_audio/liszt.wav")
    query = ft.compute_features("input_audio/kate_nash_10s.wav")
    print("df shape", query['mfcc'].shape)

    # grid_search()
    # single_search()
    # plot_genre_rand_proj()
    # plot_accuracy_rand_proj()
    # plot_genre_rand_proj()
    # random_forest()
    # predict_forest()


# **recent**
# plot_hamming_distribution()


# plot_spectral_accuracy()
# get_spectral_genre_accuracy()
# mfcc_test()
plot_genre_rand_proj()
