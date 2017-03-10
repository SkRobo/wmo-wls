#!/usr/bin/env python3
from tools import rot
import numpy as np
from scipy.sparse.linalg import lsqr
from scipy import sparse
import scipy, logging, os, multiprocessing
from tools import integr

def get_predicts(poses, indexes):
    pred = poses[indexes[:, 1]] - poses[indexes[:, 0]]
    pred[:, :2] = rot(pred[:, :2], -poses[indexes[:, 0], 2])
    return pred

def build_indices(indexes):
    l = np.sum(indexes[:, 1] - indexes[:, 0])
    indptr = np.zeros(len(indexes)+1, np.uint32)
    indices = np.zeros(l, np.uint32)
    k = 0
    for i, v in enumerate(indexes):
        n = v[1] - v[0]
        indices[k:k+n] = np.arange(v[0], v[1])
        k += n
        indptr[i+1] = k
    return indices, indptr

def build_a_xy(indexes, a_opt, indices, indptr):
    l = np.sum(indexes[:, 1] - indexes[:, 0])
    data_ang = np.zeros(l, np.float64)
    k = 0
    for i, v in enumerate(indexes):
        n = v[1] - v[0]
        data_ang[k+1:k+n] = np.cumsum(a_opt[v[0]:v[1]-1])
        k += n
    a_cos = sparse.csr_matrix((np.cos(data_ang), indices, indptr))
    a_sin = sparse.csr_matrix((np.sin(data_ang), indices, indptr))
    a = sparse.hstack([a_cos, -a_sin])
    b = sparse.hstack([a_sin, a_cos])
    return sparse.vstack([a, b], format='csr')

def filter_win(match, cov, indexes, max_win):
    mask = (indexes[:, 1] - indexes[:, 0]) <= max_win
    match = match[mask]
    cov = cov[mask]
    indexes = indexes[mask]
    return match, cov, indexes

def filter_by_odom(match, cov, indexes, odom, perc):
    if perc == 0:
        return match, cov
    ws = indexes[:, 1] - indexes[:, 0]
    od_cov = np.percentile(
        np.linalg.norm(cov[:, :2, :2], axis=(1,2))/ws, 100-perc)
    m = np.linalg.norm(cov[:, :2, :2], axis=(1,2)) > ws*od_cov
    match = match.copy()
    odom_pred = get_predicts(odom, indexes)
    match[m, :2] = odom_pred[m, :2]
    cov = cov.copy()
    cov[m, 0, 0] = od_cov
    cov[m, 1, 1] = od_cov
    return match, cov

def wls_optim(match, cov, indexes, odom, max_win=None, perc=10):
    if max_win is not None:
        match, cov, indexes = filter_win(match, cov, indexes, max_win)
    indices, indptr = build_indices(indexes)
    A = sparse.csr_matrix((np.ones(len(indices), np.float32), indices, indptr))

    cov_a = cov[:, 2, 2].copy()
    cov_a /= np.min(cov_a)
    w_a = 1/cov_a

    A_a = sparse.diags(w_a, 0, format='csr').dot(A)
    B_a = match[:, 2].copy()*w_a

    q = sparse.linalg.norm(A_a, axis=0)
    A_a = A_a.dot(sparse.diags(1/q, 0, format='csr'))

    a_opt = lsqr(A_a, B_a)[0]
    a_opt /= q

    if max_win != 1:
        match, cov = filter_by_odom(match, cov, indexes, odom, perc)

    cov_xy = np.hstack([cov[:, 0, 0], cov[:, 1, 1]])
    cov_xy /= np.min(cov_xy)
    w_xy = 1/cov_xy

    A_xy_wl = build_a_xy(indexes, a_opt, indices, indptr)
    A_xy = sparse.diags(w_xy, 0, format='csr').dot(A_xy_wl)

    B_xy = np.hstack([match[:, 0], match[:, 1]])*w_xy

    q = sparse.linalg.norm(A_xy, axis=0)
    A_xy = A_xy.dot(sparse.diags(1/q, 0, format='csr'))

    xy_opt = lsqr(A_xy, B_xy)[0]
    xy_opt /= q

    m = len(xy_opt)//2
    return np.array([xy_opt[:m], xy_opt[m:], a_opt]).T





MIT_MATCH = './results/match/mit/%s/%d.npy'
MIT_ODOMETRY = './datasets/mit/odometry/%d.npy'
MIT_OUT = './results/wls/mit/%d/'
MIT_RHO = 10

SK_MATCH = './results/match/skoltech/%s.npy'
SK_ODOMETRY = './datasets/skoltech/odometry.npy'
SK_OUT = './results/wls/skoltech/'
SK_RHO = 10

def mit_load_data(n):
    match = np.load(MIT_MATCH % ('match', n))
    cov = np.load(MIT_MATCH % ('cov', n))
    indexes = np.load(MIT_MATCH % ('indexes', n))
    odom = np.load(MIT_ODOMETRY % n)
    return match, cov, indexes, odom

def mit_worker(n):
    match, cov, indexes, odom = mit_load_data(n)
    for w in range(1, 21):
        logging.info('Processing: %d, win %d' % (n, w))
        opt = integr(wls_optim(match, cov, indexes, odom, w, MIT_RHO))
        out_path = MIT_OUT % w
        np.save(out_path + '%d.npy' % n, opt)
        logging.info('Done: %d, win %d' % (n, w))

def run_mit():
    logging.info('Started MIT')
    for w in range(1, 21):
        os.makedirs(MIT_OUT % w, exist_ok=True)

    pool = multiprocessing.Pool()
    pool.map(mit_worker, range(24))
    pool.close()
    pool.join()
    logging.info('Finished MIT')


def process(w):
    match = np.load(SK_MATCH % 'match')
    cov = np.load(SK_MATCH % 'cov')
    indexes = np.load(SK_MATCH % 'indexes')
    odom = np.load(SK_ODOMETRY)

    logging.info('Processing: win %d' % w)
    opt = integr(wls_optim(match, cov, indexes, odom, w, SK_RHO))
    np.save(SK_OUT + '%d.npy' % w, opt)
    logging.info('Done: win %d' % w)

def run_sk():
    logging.info('Started Skoltech')
    os.makedirs(SK_OUT, exist_ok=True)

    pool = multiprocessing.Pool()
    pool.map(process, range(1, 21))
    pool.close()
    pool.join()
    logging.info('Finished Skoltech')

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO)
    run_sk()
    #run_mit()
