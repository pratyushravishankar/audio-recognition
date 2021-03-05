import numpy as np
from scipy.io import loadmat
import ctypes
import logging
import argparse
import os

import features as ft
import utils
# DESCRIPTION = """


def trainSH(X, nbits):
    """
    Input
      X = features matrix[Nsamples, Nfeatures]
      param.nbits = number of bits(nbits do not need to be a multiple of 8)
    """

    [Nsamples, Ndim] = X.shape
    SHparam = {'nbits': nbits}

    # algo:
    # 1) PCA
    npca = min(nbits, Ndim)
    pc, l = eigs(np.cov(X.T), npca)
    # pc[:, 3] *= -1
    X = X.dot(pc)   # no need to remove the mean

    # 2) fit uniform distribution
    eps = np.finfo(float).eps
    mn = np.percentile(X, 5)
    mx = np.percentile(X, 95)
    mn = X.min(0) - eps
    mx = X.max(0) + eps

    # 3) enumerate eigenfunctions
    R = mx - mn
    maxMode = np.ceil((nbits+1) * R / R.max())
    nModes = int(maxMode.sum() - maxMode.size + 1)
    print("nModes", nModes)
    modes = np.ones((nModes, npca))
    m = 0
    for i in range(npca):
        # print(m + 1, m + maxMode[i])
        modes[m+1: m + int(maxMode[i]), i] = np.arange(1, int(maxMode[i])) + 1
        m = m + int(maxMode[i]) - 1
    modes = modes - 1
    omega0 = np.pi / R
    omegas = modes * omega0.values.reshape(1, -1).repeat(nModes, 0)
    eigVal = -(omegas ** 2).sum(1)
    ii = (-eigVal).argsort()
    modes = modes[ii[1:nbits+1], :]

    SHparam['pc'] = pc
    SHparam['mn'] = mn
    SHparam['mx'] = mx
    SHparam['modes'] = modes
    return SHparam


def eigs(X, npca):
    l, pc = np.linalg.eig(X)
    idx = l.argsort()[::-1][:npca]
    return pc[:, idx], l[idx]


def compressSH(X, SHparam):
    """
    [B, U] = compressSH(X, SHparam)

    Input
    X = features matrix [Nsamples, Nfeatures]
    SHparam =  parameters (output of trainSH)

    Output
    B = bits (compacted in 8 bits words)
    U = value of eigenfunctions (bits in B correspond to U>0)
    """

    if X.ndim == 1:
        X = X.reshape((1, -1))

    Nsamples, Ndim = X.shape
    nbits = SHparam['nbits']

    X = X.dot(SHparam['pc'])
    X = X - SHparam['mn']
    omega0 = np.pi / (SHparam['mx'] - SHparam['mn'])
    omegas = SHparam['modes'] * omega0.reshape((1, -1))

    U = np.zeros((Nsamples, nbits))
    for i in range(nbits):
        omegai = omegas[i, :]
        ys = np.sin(X * omegai + np.pi/2)
        yi = np.prod(ys, 1)
        U[:, i] = yi

    b = np.require(U > 0, dtype=np.int)

    print("Uncompact", b)
    B = compactbit(b)

    # return B, U
    return b


def get_hamming(X, query):

    print("prev")
    # print(X)
    for x in range(100):
        print(X[x])

    # print("query")
    # print(query)

    hammings = np.logical_xor(X, query)
    maxCount = 0

    max_id = 0
    # for h, idx in enumerate(hammings):
    #     print(h, i)
    closest = set()

    closest_k = []
    for idx, h in enumerate(hammings):
        if (h == 0).all():
            # closest.add(idx)
            closest_k.append(idx)

    #     count = 0
    #     for i in h:
    #         if i == False:
    #             count = count + 1
    #     if count == maxCount:
    #         closest.add(idx)
    #     if count > maxCount:
    #         maxCount = count
    #         # max_id = idx
    #         closest = set()
    #         closest.add(idx)
    # return closest
    return closest_k[:20]


def compactbit(b):
    nSamples, nbits = b.shape
    nwords = int((nbits + 7) / 8)
    B = np.hstack([np.packbits(b[:, i*8:(i+1)*8][:, ::-1], 1)
                   for i in range(nwords)])
    residue = nbits % 8
    if residue != 0:
        B[:, -1] = np.right_shift(B[:, -1], 8 - residue)
        print(8 - residue)

    return B


def hammingDist(B1, B2):
    print(B2.shape)
    print(B1.shape)
    print(B1)


X = utils.load("data/fma_metadata/features.csv")
# Xtest = X.iloc[0:1]
liszt = ft.compute_features("./input_audio/franz_list.mp3")
tracks = utils.load('data/fma_metadata/tracks.csv')


# print(Xtest)

sh = trainSH(X['mfcc'], 50)

B1 = compressSH(X['mfcc'], sh)
B2 = compressSH(liszt['mfcc'], sh)


# Dhamm = hammingDist(B2, B1)

closest = get_hamming(B1, B2)
print("closest ", len(closest))
print("Total ", len(B2))
print("idxes: ", closest)


top_tracks = tracks.iloc[list(closest)]
top_genres = top_tracks['track']['genre_top']

print(top_genres[top_genres.notnull()])


# print("DONE")
# print(top_genres)
