#!/usr/bin/env python3

# FMA: A Dataset For Music Analysis
# Michaël Defferrard, Kirell Benzi, Pierre Vandergheynst, Xavier Bresson, EPFL LTS2.

# All features are extracted using [librosa](https://github.com/librosa/librosa).
# Alternatives:
# * [Essentia](http://essentia.upf.edu) (C++ with Python bindings)
# * [MARSYAS](https://github.com/marsyas/marsyas) (C++ with Python bindings)
# * [RP extract](http://www.ifs.tuwien.ac.at/mir/downloads.html) (Matlab, Java, Python)
# * [jMIR jAudio](http://jmir.sourceforge.net) (Java)
# * [MIRtoolbox](https://www.jyu.fi/hum/laitokset/musiikki/en/research/coe/materials/mirtoolbox) (Matlab)

import os
import multiprocessing
import warnings
import numpy as np
from scipy import stats
import pandas as pd
import librosa
from tqdm import tqdm
import utils


def columns():
    feature_sizes = dict(chroma_stft=12, chroma_cqt=12, chroma_cens=12,
                         tonnetz=6, mfcc=20, rmse=1, zcr=1,
                         spectral_centroid=1, spectral_bandwidth=1,
                         spectral_contrast=7, spectral_rolloff=1)
    moments = ('mean', 'std', 'skew', 'kurtosis', 'median', 'min', 'max')

    columns = []
    for name, size in feature_sizes.items():
        for moment in moments:
            it = ((name, moment, '{:02d}'.format(i+1)) for i in range(size))
            columns.extend(it)

    names = ('feature', 'statistics', 'number')
    columns = pd.MultiIndex.from_tuples(columns, names=names)

    # More efficient to slice if indexes are sorted.
    return columns.sort_values()


def compute_features(path):

    features = pd.Series(index=columns(), dtype=np.float32, name="Features")

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):

        # print("Mean ",  np.mean(values, axis=1))
        # print("features[name, 'std']",  np.std(values, axis=1))
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        # print("here", tid)
        # filepath = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
        # x, sr = librosa.load(path, sr=None, mono=True)

        # read = utils.AudioreadLoader(44100)
        # res = read._load(path)

        # print("audioread", res)

        # print("?>>> ", x, sr)

        x, sr = librosa.load(path, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(
            x, frame_length=2048, hop_length=512)

        # print("FFFFFF ", type(x), x)

        # print(f)

        feature_stats('zcr', f)

        # print("AFTEER ")

        # cqt = np.abs(librosa.cqt(x, sr=sr, hop_length=512, bins_per_octave=12,
        #                          n_bins=7*12, tuning=None))

        # print("HERE")
        # assert cqt.shape[0] == 7 * 12
        # assert np.ceil(len(x)/512) <= cqt.shape[1] <= np.ceil(len(x)/512)+1

        # f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        # feature_stats('chroma_cqt', f)
        # f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        # feature_stats('chroma_cens', f)
        # f = librosa.feature.tonnetz(chroma=f)
        # feature_stats('tonnetz', f)

        # del cqt

        # print("HERE")
        stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))
        # print("stft", stft)
        # assert stft.shape[0] == 1 + 2048 // 2
        # assert np.ceil(len(x)/512) <= stft.shape[1] <= np.ceil(len(x)/512)+1
        # del x

        # f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        # feature_stats('chroma_stft', f)

        # f = librosa.feature.rmse(S=stft)
        # feature_stats('rmse', f)

        # f = librosa.feature.spectral_centroid(S=stft)
        # feature_stats('spectral_centroid', f)
        # f = librosa.feature.spectral_bandwidth(S=stft)
        # feature_stats('spectral_bandwidth', f)
        # f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        # feature_stats('spectral_contrast', f)
        # f = librosa.feature.spectral_rolloff(S=stft)
        # feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)
        # del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

        print(" mfccssss ")
        feature_stats('mfcc', f)

    except Exception as e:
        import traceback
        print("CAUGHT")
        # print('{}: {}'.format("5", repr(e)))
        print(traceback.format_exc())

    return features.to_frame().transpose()


def compute_microphone_features(stream):

    features = pd.Series(index=columns(), dtype=np.float32, name="Features")

    # Catch warnings as exceptions (audioread leaks file descriptors).
    warnings.filterwarnings('error', module='librosa')

    def feature_stats(name, values):
        features[name, 'mean'] = np.mean(values, axis=1)
        features[name, 'std'] = np.std(values, axis=1)
        features[name, 'skew'] = stats.skew(values, axis=1)
        features[name, 'kurtosis'] = stats.kurtosis(values, axis=1)
        features[name, 'median'] = np.median(values, axis=1)
        features[name, 'min'] = np.min(values, axis=1)
        features[name, 'max'] = np.max(values, axis=1)

    try:
        # print("here", tid)
        # filepath = utils.get_audio_path(os.environ.get('AUDIO_DIR'), tid)
        # x, sr = librosa.load(stream, sr=None, mono=True)

        # read = utils.AudioreadLoader(SAMPLING_RATE=44100)
        # x, sr = librosa.load(librosa.util.example_audio_file(), duration=5.0)

        # print("x ", x)
        # print("sr ", sr)

        # x, sr = librosa.load(filepath, sr=None, mono=True)  # kaiser_fast

        f = librosa.feature.zero_crossing_rate(
            stream, frame_length=2048, hop_length=512)
        feature_stats('zcr', f)

        cqt = np.abs(librosa.cqt(stream, sr=44100, hop_length=512, bins_per_octave=12,
                                 n_bins=7*12, tuning=None))
        assert cqt.shape[0] == 7 * 12
        assert np.ceil(
            len(stream)/512) <= cqt.shape[1] <= np.ceil(len(stream)/512)+1

        f = librosa.feature.chroma_cqt(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cqt', f)
        f = librosa.feature.chroma_cens(C=cqt, n_chroma=12, n_octaves=7)
        feature_stats('chroma_cens', f)
        f = librosa.feature.tonnetz(chroma=f)
        feature_stats('tonnetz', f)

        del cqt
        stft = np.abs(librosa.stft(stream, n_fft=2048, hop_length=512))
        assert stft.shape[0] == 1 + 2048 // 2
        assert np.ceil(
            len(stream)/512) <= stft.shape[1] <= np.ceil(len(stream)/512)+1
        del stream

        f = librosa.feature.chroma_stft(S=stft**2, n_chroma=12)
        feature_stats('chroma_stft', f)

        f = librosa.feature.rmse(S=stft)
        feature_stats('rmse', f)

        f = librosa.feature.spectral_centroid(S=stft)
        feature_stats('spectral_centroid', f)
        f = librosa.feature.spectral_bandwidth(S=stft)
        feature_stats('spectral_bandwidth', f)
        f = librosa.feature.spectral_contrast(S=stft, n_bands=6)
        feature_stats('spectral_contrast', f)
        f = librosa.feature.spectral_rolloff(S=stft)
        feature_stats('spectral_rolloff', f)

        mel = librosa.feature.melspectrogram(sr=44100, S=stft**2)
        del stft
        f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
        feature_stats('mfcc', f)

    except Exception as e:
        # print(e)
        import sys
        # print('{}: {}'.format("5", repr(e)))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(">>>> ", exc_type, fname, exc_tb.tb_lineno)

    return features.to_frame().transpose()


def main():
    tracks = utils.load('data/fma_metadata/tracks.csv')
    features = pd.DataFrame(index=tracks.index,
                            columns=columns(), dtype=np.float32)

    # More than usable CPUs to be CPU bound, not I/O bound. Beware memory.
    # nb_workers = int(1.5 * len(os.sched_getaffinity(0)))
    nb_workers = int(1.5 * os.cpu_count())

    # Longest is ~11,000 seconds. Limit processes to avoid memory errors.
    # table = ((5000, 1), (3000, 3), (2000, 5), (1000, 10), (0, nb_workers))
    # for duration, nb_workers in table:
    #     print('Working with {} processes.'.format(nb_workers))

    #     tids = tracks[tracks['track', 'duration'] >= duration].index
    #     print("TIDS: ", tids)
    #     tracks.drop(tids, axis=0, inplace=True)

    #     pool = multiprocessing.Pool(nb_workers)
    #     it = pool.imap_unordered(compute_features, tids)

    #     for i, row in enumerate(tqdm(it, total=len(tids))):
    #         features.loc[row.name] = row

    #         if i % 1000 == 0:
    #             save(features, 10)

    # save(features, 10)
    # test(features, 10)

    # print(compute_features(2).shape)
    # print(compute_features(2))


def save(features, ndigits):

    # Should be done already, just to be sure.
    features.sort_index(axis=0, inplace=True)
    features.sort_index(axis=1, inplace=True)

    features.to_csv('features.csv', float_format='%.{}e'.format(ndigits))


def test(features, ndigits):

    indices = features[features.isnull().any(axis=1)].index
    if len(indices) > 0:
        print('Failed tracks: {}'.format(', '.join(str(i) for i in indices)))

    tmp = utils.load('features.csv')
    np.testing.assert_allclose(tmp.values, features.values, rtol=10**-ndigits)


if __name__ == "__main__":
    main()


# liszt = compute_features(
    # "/Users/pratyush/Documents/Uni/cs310/code/fma/input_audio")


# liszt = compute_features("input_audio/franz_list.mp3")

# print("LK JSA", liszt)

# y, sr = librosa.load(librosa.util.example_audio_file(), duration=5.0)

# y, sr = librosa.load("/Users/pratyush/Documents/Uni/cs310/code/fma/output.wav")
# print("y and sr: ", y, sr)

# res = compute_features(
#     "/Users/pratyush/Documents/Uni/cs310/code/fma/output.wav")
# print("res ", res['mfcc'])

# print(y, sr)

# print(">>> ", liszt['zcr']['max'
