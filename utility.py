import numpy as np
# from scipy.spatial.distance import cdist
import time
from multiprocessing import Pool
import os, random
import torch
import torch.utils
from torch.autograd import Variable


# from scipy.spatial.distance import cosine
# other
def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def calc_MAP(array2d, version, que_range=None, K=1e10):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0

    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):

            if version[u] == version[v]:

                if k < K:
                    version_cnt += 1
                    per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)


def calc_MAP(array2d, version, que_range=None, K=1e10):
    if que_range is not None:
        que_s, que_t = que_range[0], que_range[1]
        if que_s == 0:
            ref_s, ref_t = que_t, len(array2d)
        else:
            ref_s, ref_t = 0, que_s
    else:
        que_s, que_t, ref_s, ref_t = 0, len(array2d), 0, len(array2d)

    new_array2d = []
    for u, row in enumerate(array2d[que_s: que_t]):
        row = [(v + ref_s, col) for (v, col) in enumerate(row[ref_s: ref_t]) if u + que_s != v + ref_s]
        new_array2d.append(row)
    MAP, top10, rank1 = 0, 0, 0

    for u, row in enumerate(new_array2d):
        row.sort(key=lambda x: x[1])
        per_top10, per_rank1, per_MAP = 0, 0, 0
        version_cnt = 0.
        u = u + que_s
        for k, (v, val) in enumerate(row):

            if version[u] == version[v]:

                if k < K:
                    version_cnt += 1
                    per_MAP += version_cnt / (k + 1)
                if per_rank1 == 0:
                    per_rank1 = k + 1
                if k < 10:
                    per_top10 += 1
        per_MAP /= 1 if version_cnt < 0.0001 else version_cnt
        # if per_MAP < 0.1:
        #     print row
        MAP += per_MAP
        top10 += per_top10
        rank1 += per_rank1
    return MAP / float(que_t - que_s), top10 / float(que_t - que_s) / 10, rank1 / float(que_t - que_s)


def get_dis2d4(seqs, verbose=False):
    start_time = time.time()
    dis2d = np.zeros((len(seqs), len(seqs)))
    for i, seq1 in enumerate(seqs):
        idx = np.where(seq1 != 0)
        x = seq1[idx].squeeze()
        for j, seq2 in enumerate(seqs):
            y = seq2[idx].squeeze()
            dis2d[i][j] = 1 - np.dot(x, y)
    end_time = time.time()
    if verbose:
        print('time: %fs' % (end_time - start_time))
    return dis2d


def get_disEu(seqs, verbose=False):
    start_time = time.time()
    dis2d = np.zeros((len(seqs), len(seqs)))
    for i, seq1 in enumerate(seqs):
        idx = np.where(seq1 != 0)
        x = seq1[idx].squeeze()
        for j, seq2 in enumerate(seqs):
            y = seq2[idx].squeeze()
            dis2d[i][j] = np.sqrt(np.sum(np.square(x - y)))
    end_time = time.time()
    if verbose:
        print('time: %fs' % (end_time - start_time))
    return dis2d



