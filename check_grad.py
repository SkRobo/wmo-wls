#!/usr/bin/env python3
import numpy as np
from wls import build_indices, build_a_xy, mit_load_data, wls_optim
from scipy import sparse
from scipy.sparse.linalg import lsqr
from datetime import datetime
import scipy
import matplotlib.pyplot as plt
from nonlinear import comp_jacob, f, gen_masks


def grad(s, m, w_xy, w_phi, alpha, indexes, indices, indptr, masks):
    l = len(s)//3
    s_xy = s[:2*l]
    s_phi = s[2*l:]

    P = build_a_xy(indexes, s_phi, indices, indptr)
    P = sparse.diags(w_xy, 0, format='csr').dot(P)

    A = sparse.csr_matrix((np.ones(len(indices), np.float32), indices, indptr))
    A = sparse.diags(w_phi, 0, format='csr').dot(A)

    j = comp_jacob(s, m, w_xy, A, P, alpha, indexes, masks)
    return j


if __name__ == '__main__':
    match, cov, indexes, odom = mit_load_data(1)
    m = (indexes[:, 1] <= 800) & (indexes[:, 0] >= 700)
    match = match[m]
    cov = cov[m]
    indexes = indexes[m]
    indexes -= np.min(indexes)

    sort_ind = np.lexsort((-1 - indexes[:, 0], indexes[:, 1]))
    match = match[sort_ind]
    cov = cov[sort_ind]
    indexes = indexes[sort_ind]

    d0 = wls_optim(match, cov, indexes, None, perc=0)

    s0 = np.hstack([d0[:, 0], d0[:, 1], d0[:, 2]])
    w_xy = np.hstack([
        1/cov[:, 0, 0],
        1/cov[:, 1, 1],
    ])
    w_phi = 1/cov[:, 2, 2]
    m = np.hstack([
        match[:, 0]/cov[:, 0, 0],
        match[:, 1]/cov[:, 1, 1],
        match[:, 2]/cov[:, 2, 2],
    ])
    indices, indptr = build_indices(indexes)

    alpha = 1

    df = np.zeros(s0.shape)
    dxy = 0.0000000001
    dphi = 0.0000000000001
    l = len(s0)//3
    for i in range(3*l):
        if i < 2*l:
            dd = dxy
        else:
            dd = dphi

        sc = s0.astype(np.float64)
        sc[i] += dd
        fv1 = f(sc, m, w_xy, w_phi, alpha, indexes, indices, indptr)
        sc = s0.astype(np.float64)
        sc[i] -= dd
        fv2 = f(sc, m, w_xy, w_phi, alpha, indexes, indices, indptr)
        df[i] = (fv1 - fv2)/(2*dd)

    masks = gen_masks(indexes)
    j0 = grad(s0.astype(np.float64), m, w_xy, w_phi, alpha,
        indexes, indices, indptr, masks)

    plt.plot(j0)
    plt.plot(df)
    #print(df)
    plt.show()
