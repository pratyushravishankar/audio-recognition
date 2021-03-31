import pyaudio
import wave
import utils
import sys
import pickle
import time
import features as ft
import heapq
from sklearn.model_selection import train_test_split
import evaluation as Eval
import lsh_random_projection as LSH
import spectral_hashing as Spectral


def build_model():

    features = utils.load("data/fma_metadata/features.csv")

    X_train, X_test = train_test_split(features, test_size=1)
    lsh = LSH.LSH(1, 15, 140)
    lsh.add(X_train['mfcc'])
    # eval = Eval.Evaluation(lsh)

    sh = Spectral.trainSH(X_train['mfcc'], 200)
    B1 = Spectral.compressSH(X_train['mfcc'], sh)
    # print("DONE sh, b1")

    objList = [lsh, features, B1, sh]


# B2 = compressSH(liszt['mfcc'], sh)

    save_object(objList, 'lsh.pkl')


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def outputGenre(eval, features):

    print("output genre HERE")

    X_train, X_test = train_test_split(features, test_size=1)
    # lsh = LSH.LSH(1, 15, 140)
    # lsh.add(X_train['mfcc'])
    query = features.iloc[1:2]

    # top_k = lsh.get(
    # features['mfcc'], query['mfcc'], probeType="bit-flip")

    print("GETTING")
    top_k = eval.lsh.get(inp_vec=query['mfcc'], probeType="step-wise")

    genre = get_top_genre(top_k[0])

    print("genre", genre)


# sample usage


def get_top_genre(top_k):

    print(top_k)

    genre_count = {}
    # print("items")
    for item in top_k['genre']:
        # print(item)
        if item not in genre_count:
            genre_count[item] = 0
        genre_count[item] += 1

    curMax = 0
    maxGenre = ""
    for genre in genre_count:
        # curMax = max(genre_count[genre], curMax)
        if genre_count[genre] > curMax:
            curMax = genre_count[genre]
            maxGenre = genre

    return maxGenre


def eval_top_k_accuracy(eval, features, B1, sh):

    X_train, X_test = train_test_split(features, test_size=1)

    # query = ft.compute_features("input_audio/26 Queensway 4.wav")

    # query = features.iloc[1:2]
    query = ft.compute_features("input_audio/liszt.wav")

    brute_force_top_k = eval.bruteforce_get(
        X_train['mfcc'], query['mfcc'])

    lsh_random_proj_top_k = eval.lsh.get(
        query['mfcc'], probeType="rand_proj")

    lsh_probe_step_wise_top_k = eval.lsh.get(
        query['mfcc'], probeType="step-wise")

    lsh_probe_bit_flip_top_k = eval.lsh.get(
        query['mfcc'], probeType="bit-flip")

    # B2 = Spectral.compressSH(query['mfcc'], sh)
    # closest = Spectral.get_hamming(B1, B2)

    # tracks = utils.load('data/fma_metadata/tracks.csv')

    # top_tracks = tracks.iloc[list(closest)]
    # top_genres = top_tracks['track']['genre_top']

    # spectral_top_k = top_genres[top_genres.notnull()]

    # with open('out.txt', 'w') as f:

    # f = open('out.txt','w')
    with open('out.txt', 'w') as f:

        # print("BRUETY ", brute_force_top_k)
        print(brute_force_top_k, file=f)
        print(lsh_random_proj_top_k, file=f)
        print(lsh_probe_step_wise_top_k, file=f)
        print(lsh_probe_bit_flip_top_k, file=f)
        print(spectral_top_k, file=f)

    # print("RAND PROJ ", lsh_random_proj_top_k)
    # print("STEP-wise ", lsh_probe_step_wise_top_k)
    # print("bit flip ", lsh_probe_bit_flip_top_k)
    # print("spectral ", spectral_top_k)


# outputGenre()
# build_model()

def get_features(song, features):

    if song == "null":

        print("LIST")
        CHUNK = 2048
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100
        RECORD_SECONDS = 10
        WAVE_OUTPUT_FILENAME = "output.wav"

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

        return ft.compute_features('./input_audio/liszt.wav')
    else:
        return features.iloc[0:1]


def test(type, song):

    print("Loading features")
    features = utils.load("data/fma_metadata/features.csv")
    # X_train, X_test = train_test_split(features[1:])

    X_train, X_test = train_test_split(
        features, test_size=1, random_state=42)

    print("Populating hashmaps..")
    lsh = LSH.LSH(1, 15, 140)
    # lsh.add(X_train['mfcc'], True)

    # query = ft.compute_features("input_audio/26 Queensway 4.wav")
    # query = ft.compute_features("output.wav")

    # query = get_features(song, features)

    # query = ft.compute_features(song, features)
    if type == "-demo":

        tic = time.perf_counter()
        # res = lsh.get(query['mfcc'], probeType="bit-flip")
        res = lsh.get(query['mfcc'], probeType="bit-flip")

        top = get_top_genre(res[0])

        toc = time.perf_counter()

        with open('out.txt', 'w') as f:

            print(res, file=f)

            print(f"time: {toc - tic:0.4f} seconds", file=f)

            print("Top genre", top, file=f)

    else:

        print("START")
        # proj = lsh.get(query['mfcc'], probeType="rand_proj")
        # flip = lsh.get(query['mfcc'], probeType="bit-flip")
        # step = lsh.get(query['mfcc'], probeType="step-wise")
        eval = Eval.Evaluation(lsh)
        # brute_force = eval.bruteforce_get(
        #     X_train['mfcc'], query['mfcc'])
        brute_force = eval.bruteforce_get(
            X_train['mfcc'], X_test['mfcc'])

        forest = Eval.predict_forest(X_test['mfcc'])

        sh = Spectral.trainSH(X_train['mfcc'], 100)

        B2 = Spectral.compressSH(X_train['mfcc'], sh)
        B1 = Spectral.compressSH(X_test['mfcc'], sh)

        results = Spectral.hammingDist(B1, B2)

        k = 20

        first_idx, pd = Eval.get_hamming_dist(results[0])

        top_k_queries = []
        for query in results:

            # query = sort(query)
            # print("query :", query)
            # smallest = heapq.nsmallest(k, query)
            smallest = sorted(range(len(query)), key=lambda k: query[k])
            top_k_queries.append(smallest[:k])

            # while (query < )

        # print("top_K : ", top_k)
        closest = top_k_queries[0]
        print("closest trial : ", closest)
        print("first idx: ", first_idx)
        print("pd : ", pd)

        # closest = Spectral.get_hamming(B1, B2)

        tracks = utils.load('data/fma_metadata/tracks.csv')

        top_tracks = tracks.iloc[list(closest)]
        top_genres = top_tracks['track']['genre_top']

        spectral_top_k = top_genres[top_genres.notnull()]

        # print("WREITE")
        with open('out.txt', 'w') as f:

            print("RANDOM_PROJECTION", file=f)
            print(proj, file=f)
            print("BIT-FLIP HASHING", file=f)
            print(flip, file=f)
            print("STEP-WISE HASHING", file=f)
            print(step, file=f)
            print("SPECTRAL HASHING", file=f)
            print(spectral_top_k.to_string(), file=f)
            print("BRUTEFORCE", file=f)
            print(brute_force, file=f)
            print("\n forest prediction", file=f)
            print(forest, file=f)


def main(argv):
    # if
    print(argv)

    # with open('lsh.pkl', 'rb') as input:
    #     obj = pickle.load(input)
    #     eval = obj[0]
    #     features = obj[1]
    #     B1 = obj[2]
    #     sh = obj[3]

    # if (argv[0] == '-add'):

    #     print("ADDING")
    #     build_model()
    #     # eval_top_k_accuracy(eval, features)
    # elif (argv[0] == "-demo"):
    #     eval_top_k_accuracy(eval, features, B1, sh)
    # else:
    #     res = outputGenre(eval, features)
    #     print(res)

    test(argv[0], argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])
