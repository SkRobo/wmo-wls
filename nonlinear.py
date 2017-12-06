#!/usr/bin/env python3
from wls import build_indices, build_a_xy, wls_optim, mit_load_data
from wls import filter_by_odom
from tools import poses_to_zero, integr
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import lsqr
import scipy, random
import sys, os, logging, multiprocessing, progressbar

ALPHAS = [
    0.001, 0.025, 0.005, 0.0075,
    0.01, 0.025, 0.05, 0.075,
    0.1, 0.5, 1.0,
]

def calc_p_deriv_i(i, s_phi, indexes, masks):
    k = 0
    # precomputed value for max_win=20
    l = 1330
    angles = np.empty(l, np.float32)
    row_ind = np.empty(l, np.uint32)
    column_ind = np.empty(l, np.uint32)

    old_v1 = None
    for m in masks[i]:
        v1 = indexes[m][1]
        if old_v1 is None or old_v1 != v1:
            angs = s_phi[i:v1-1]
            n = len(angs)
            ang_cum = np.cumsum(angs)
            inds = np.arange(i+1, v1)
            old_v1 = v1

        angles[k:k+n] = ang_cum
        row_ind[k:k+n] = m*np.ones(n)
        column_ind[k:k+n] = inds
        k += n

    angles = angles[:k]
    row_ind = row_ind[:k]
    column_ind = column_ind[:k]

    shape = (len(indexes), len(s_phi))
    a_cos = sparse.coo_matrix((angles, (row_ind, column_ind)), shape)
    a_sin = a_cos.copy()
    a_cos.data = np.cos(a_cos.data)
    a_sin.data = np.sin(a_sin.data)

    a = sparse.hstack([-a_sin, -a_cos], format='coo')
    b = sparse.hstack([a_cos, -a_sin], format='coo')
    return sparse.vstack([a, b], format='coo')


def calc_p_deriv(s_xy, s_phi, w_xy, r_xy, e_xy, indexes, masks):
    n = len(s_phi)
    p_deriv = np.zeros(n, np.float32)
    sparse_w_xy = sparse.diags(w_xy, 0, format='csr')
    for i in range(n-1):
        deriv = calc_p_deriv_i(i, s_phi, indexes, masks)
        p_deriv[i] = (sparse_w_xy.dot(deriv).dot(s_xy)).T.dot(r_xy)
    return 1/e_xy*p_deriv


def comp_jacob(x, m, w_xy, A, P, alpha, indexes, masks):
    n = len(x)//3
    assert(len(x)%3 == 0)
    s_xy = x[:2*n]
    s_phi = x[-n:]

    n2 = len(m)//3
    m_xy = m[:2*n2]
    m_phi = m[-n2:]

    r_xy = P.dot(s_xy) - m_xy
    r_phi = A.dot(s_phi) - m_phi

    e_xy = np.linalg.norm(r_xy)
    e_phi = np.linalg.norm(r_phi)

    j_xy = P.T.dot(r_xy)/e_xy

    j_phi = alpha/e_phi*A.T.dot(r_phi)

    j_phi += calc_p_deriv(s_xy, s_phi, w_xy, r_xy, e_xy, indexes, masks)
    return np.hstack([j_xy, j_phi])


def f(s, m, w_xy, w_phi, alpha, indexes, indices, indptr, masks=None):
    l = len(s)//3
    s_xy = s[:2*l]
    s_phi = s[2*l:]

    l2 = len(m)//3
    m_xy = m[:2*l2]
    m_phi = m[2*l2:]

    P = build_a_xy(indexes, s_phi, indices, indptr)
    P = sparse.diags(w_xy, 0, format='csr').dot(P)

    A = sparse.csr_matrix((np.ones(len(indices), np.float32), indices, indptr))
    A = sparse.diags(w_phi, 0, format='csr').dot(A)

    r_xy = (P.dot(s_xy) - m_xy).astype(np.float64)
    r_phi = (A.dot(s_phi) - m_phi).astype(np.float64)
    e = np.linalg.norm(r_xy) + alpha*np.linalg.norm(r_phi)
    logging.debug('Calculated error function: %f', e)
    return e

def grad(s, m, w_xy, w_phi, alpha, indexes, indices, indptr, masks):
    logging.debug('Jacobian start: %.3f' % alpha)
    l = len(s)//3
    s_xy = s[:2*l]
    s_phi = s[2*l:]

    P = build_a_xy(indexes, s_phi, indices, indptr)
    P = sparse.diags(w_xy, 0, format='csr').dot(P)

    A = sparse.csr_matrix((np.ones(len(indices), np.float32), indices, indptr))
    A = sparse.diags(w_phi, 0, format='csr').dot(A)

    j = comp_jacob(s, m, w_xy, A, P, alpha, indexes, masks)
    logging.debug('Jacobian done: %.3f' % alpha)
    return j

def get_args(match, cov, indexes, odom, perc):
    match, cov = filter_by_odom(match, cov, indexes, odom, perc)
    w_phi = 1/cov[:, 2, 2]
    m_phi = match[:, 2]*w_phi
    w_xy = 1/np.hstack([cov[:, 0, 0], cov[:, 1, 1]])
    m_xy = np.hstack([match[:, 0], match[:, 1]])*w_xy
    m = np.hstack([m_xy, m_phi])
    indices, indptr = build_indices(indexes)
    return (m, w_xy, w_phi, indexes, indices, indptr)

def gen_masks(indexes):
    masks = [
        np.nonzero((indexes[:, 0] <= i) & (i < indexes[:, 1]))[0]
        for i in range(np.max(indexes[:, 0]))
    ]
    for i, v in enumerate(masks):
        masks[i] = v[indexes[v, 1] - 1 > i]
    return masks


def nonlin_optim(s0, alpha, args):
    m, w_xy, w_phi, indexes, indices, indptr = args
    masks = gen_masks(indexes)
    return scipy.optimize.minimize(f, s0, jac=grad, method='BFGS',
        args=(m, w_xy, w_phi, alpha, indexes, indices, indptr, masks))

RHO = 10


def worker(args):
    dataset_n, start_index, end_index, queue = args
    os.makedirs('./results/nonlinear/%d/' % dataset_n, exist_ok=True)

    match, cov, indexes, odom = mit_load_data(dataset_n)

    m = np.ones(len(indexes), np.bool)
    if start_index is not None:
        m &= indexes[:, 0] >= start_index

    if end_index is not None:
        m &= indexes[:, 1] <= end_index

    match = match[m]
    cov = cov[m]
    indexes = indexes[m]
    indexes -= np.min(indexes)

    logging.info('---- Linear optimization n: %d, start: %d',
            dataset_n, start_index)
    out_path = 'results/nonlinear/%d/%d_linear.npy' % (
        dataset_n, start_index)
    if os.path.exists(out_path):
        logging.info('---- Reusing linear optimization n: %d, start: %d',
            dataset_n, start_index)
        d0 = np.load(out_path)
    else:
        d0 = wls_optim(match, cov, indexes, odom, perc=RHO).astype(np.float32)
        np.save(out_path, d0)

    d0 = np.hstack([d0[:, 0], d0[:, 1], d0[:, 2]])

    s0 = d0.copy()

    args = get_args(match, cov, indexes, odom, RHO)

    (m, w_xy, w_phi, indexes, indices, indptr) = args

    for alpha in reversed(ALPHAS):
        logging.info('==== Started n: %d, start: %d, alpha: %.3f',
            dataset_n, start_index, alpha)
        out_path = 'results/nonlinear/%d/%d_%.3f.npy' % (
            dataset_n, start_index, alpha)
        if os.path.exists(out_path):
            d = np.load(out_path)
            s0 = np.hstack([d[:, 0], d[:, 1], d[:, 2]])
            logging.info('==== Reusing n: %d, start: %d, alpha: %.3f',
                dataset_n, start_index, alpha)
            continue
        lin = f(d0, m, w_xy, w_phi, alpha, indexes, indices, indptr)
        init = f(s0, m, w_xy, w_phi, alpha, indexes, indices, indptr)

        res = nonlin_optim(s0, alpha, args)

        logging.info('==== Finished n: %d, start: %d, alpha: %.3f',
            dataset_n, start_index, alpha)
        logging.info('Linear:\t%f', lin)
        logging.info('Init:\t%f', init)
        logging.info('After:\t%f', res.fun)

        l = len(res.x)//3
        res_data = np.array([res.x[:l], res.x[l:-l], res.x[-l:]]).T

        np.save(out_path, res_data)
        s0 = res.x

        queue.put(None)


BLOCK = 1200
STEP = 600

if __name__ == '__main__':
    #logging.basicConfig(
    #    format='[%(asctime)s] %(levelname)s: %(message)s',
    #    level=logging.INFO)

    tasks = []
    pool = multiprocessing.Pool()
    queue = multiprocessing.Manager().Queue()
    for dataset_n in range(24):
        gt = np.load('datasets/mit/ground_truth/%d.npy' % dataset_n)
        k = 0
        while k < len(gt) - BLOCK//2:
            start = k
            end = k+BLOCK
            task = (dataset_n, start, k+BLOCK, queue)
            tasks.append(task)
            k += STEP

    random.shuffle(tasks)

    pool.imap(worker, tasks, chunksize=1)
    bar = progressbar.ProgressBar()
    for _ in bar(range(len(tasks)*len(ALPHAS))):
        queue.get()
    pool.close()
    pool.join()

