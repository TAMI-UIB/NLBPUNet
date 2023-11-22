# -*- coding: utf-8 -*-
"""
Created on 2020/1/14 11:58

@author: Evan Chen
"""
import pickle
import cv2 as cv
import numpy as np
from math import floor
from scipy.io import loadmat, savemat
from itertools import product

def erags(MS, F):
    MS = MS.astype(np.float64)
    F = F.astype(np.float64)

    m, n, p = F.shape
    mn = n * m
    A1 = F[:, :, 0].reshape(mn, 1)
    A2 = F[:, :, 1].reshape(mn, 1)
    A3 = F[:, :, 2].reshape(mn, 1)
    A4 = F[:, :, 3].reshape(mn, 1)

    C1 = np.sqrt(np.sum(np.square(MS[:, :, 0] - F[:, :, 0])) / mn)
    C2 = np.sqrt(np.sum(np.square(MS[:, :, 1] - F[:, :, 1])) / mn)
    C3 = np.sqrt(np.sum(np.square(MS[:, :, 2] - F[:, :, 2])) / mn)
    C4 = np.sqrt(np.sum(np.square(MS[:, :, 3] - F[:, :, 3])) / mn)

    S1 = np.square((C1 / np.mean(A1, axis=0)))
    S2 = np.square((C2 / np.mean(A2, axis=0)))
    S3 = np.square((C3 / np.mean(A3, axis=0)))
    S4 = np.square((C4 / np.mean(A4, axis=0)))

    S = S1 + S2 + S3 + S4
    N = pow(S / 4, 0.5) * 25
    return N[0]


def sam(MS, F):
    MS = MS.astype(np.float64)
    F = F.astype(np.float64)

    m, n, p = F.shape
    A, M = np.zeros_like(F), np.zeros_like(F)
    for i in range(p):
        A[:, :, i] = MS[:, :, i] * F[:, :, i]
        M[:, :, i] = np.square(MS[:, :, i])
        F[:, :, i] = np.square(F[:, :, i])

    Asum = np.sum(A, axis=2)
    Msum = np.sqrt(np.sum(M, axis=2))
    Fsum = np.sqrt(np.sum(F, axis=2))

    U = np.zeros_like(Msum)
    low = Msum * Fsum

    for i in range(m):
        for j in range(n):

            if low[i, j] == 0:
                U[i, j] = Asum[i, j] / 1e-8
            else:
                U[i, j] = Asum[i, j] / low[i, j]

            if U[i, j] > 1:
                U[i, j] = 1

    angle = np.sum(np.degrees(np.arccos(U))) / (m * n)
    return angle


def q4(MS, F):
    a1, b1, c1, d1 = MS[:, :, 0], MS[:, :, 1], MS[:, :, 2], MS[:, :3]
    a2, b2, c2, d2 = F[:, :, 0], F[:, :, 1], F[:, :, 2], F[:, :3]

    nrows, ncols = a1.size()
    nrows_out = floor(nrows / 8)
    ncols_out = floor(ncols / 8)
    q4_map = np.zeros((nrows_out, ncols_out))

    for i in range(nrows_out):
        for j in range(ncols_out):
            crows_low, ccols_low = 1 + 8 * i, 1 + 8 * j
            crows_up, ccols_up = 8 + 8 * i + 1, 8 + 8 * j + 1

            sa1 = a1[crows_low:crows_up, ccols_low, ccols_up].ravel()
            sb1 = b1[crows_low:crows_up, ccols_low, ccols_up].ravel()
            sc1 = c1[crows_low:crows_up, ccols_low, ccols_up].ravel()
            sd1 = d1[crows_low:crows_up, ccols_low, ccols_up].ravel()

            sa2 = a2[crows_low:crows_up, ccols_low, ccols_up].ravel()
            sb2 = b2[crows_low:crows_up, ccols_low, ccols_up].ravel()
            sc2 = c2[crows_low:crows_up, ccols_low, ccols_up].ravel()
            sd2 = d2[crows_low:crows_up, ccols_low, ccols_up].ravel()

            pass

def caculte_assessment(test_img):
    hrms = loadmat('resource/record_9.mat')['record'][:, :, :-1]

    r, c, _ = test_img.shape
    hrms = hrms[:r, :c, :]

    #test_img[test_img < 0] = 0
    print('sam:{}   ergas:{}'.format(sam(hrms, test_img), erags(hrms, test_img)))


    block_size = 64
    row, col, _ = test_img.shape
    max_r = (row // block_size) * block_size
    max_c = (col // block_size) * block_size
    test_max_c = max_c  # 鐢ㄤ簬杩樺師鍥惧儚

    x = [r_ix for r_ix in range(0, max_r, block_size)]
    y = [c_ix for c_ix in range(0, max_c, block_size)]
    test_block_coord = list(product(x, y))

    blocks = []
    for r, c in test_block_coord:
        dr = r + block_size
        dc = c + block_size
        test_block = test_img[r:dr, c:dc, :]
        label_block = hrms[r:dr, c:dc, :]
        blocks.append([label_block, test_block])

    score = [(sam(*item), erags(*item)) for item in blocks]

    # sam = np.array(sorted([x[0] for x in score]))
    # erags = np.array(sorted(x[1] for x in score))
    sam_score = np.array([x[0] for x in score])
    erags_score = np.array([x[1] for x in score])

    print('min, max, mean, median:')
    print(sam_score.min(), sam_score.max(), sam_score.mean(), np.median(sam_score))
    print(erags_score.min(), erags_score.max(), erags_score.mean(), np.median(erags_score))

if __name__ == '__main__':
    # 瑙嗚璇勪环
    # from scipy.io import loadmat
    # mat = loadmat('resource/record_9.mat')['record']
    # cv.imshow('', mat[:,:,-1])
    # cv.waitKey(0)

    from scipy.io import loadmat, savemat

    hrms = loadmat('../resource/wv/record_9.mat')['record'][:, :, :-1]

    with open('../cache/img.pickle', 'rb') as fb:
        test_img = pickle.load(fb)
        r, c, _ = test_img.shape
        hrms = hrms[:r, :c, :]

        test_img[test_img < 0] = 0
        print('sam:{}   ergas:{}'.format(sam(hrms, test_img), erags(hrms, test_img)))

        #savemat(r'C:\Users\Administrator\Desktop\hrms_img.mat', {'hrms': hrms})
        #savemat(r'C:\Users\Administrator\Desktop\pred_img.mat', {'pred_img': test_img})

        #cv.imshow('', test_img[:, :, :3])
        #cv.waitKey(0)
        block_size = 64
        row, col, _ = test_img.shape
        max_r = (row // block_size) *block_size
        max_c = (col // block_size) * block_size
        test_max_c = max_c  # 鐢ㄤ簬杩樺師鍥惧儚

        x = [r_ix for r_ix in range(0, max_r, block_size)]
        y = [c_ix for c_ix in range(0, max_c, block_size)]
        test_block_coord = list(product(x, y))

        blocks = []
        for r, c in test_block_coord:
            dr = r + block_size
            dc = c + block_size
            test_block = test_img[r:dr, c:dc, :]
            label_block = hrms[r:dr, c:dc, :]
            blocks.append([label_block, test_block])

        score = [(sam(*item), erags(*item)) for item in blocks]

        #sam = np.array(sorted([x[0] for x in score]))
        #erags = np.array(sorted(x[1] for x in score))
        sam = np.array([x[0] for x in score])
        erags = np.array([x[1] for x in score])

        print(sam.min(), sam.max(), sam.mean(), np.quantile(sam, 0.5))
        print(erags.min(), erags.max(), erags.mean(), np.quantile(erags, 0.5))

        from matplotlib import pyplot as plt
        plt.plot(sam)
        plt.plot(erags)
        plt.show()
