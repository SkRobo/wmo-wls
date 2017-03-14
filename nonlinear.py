#!/usr/bin/env python3
from wls import build_indices, build_a_xy, wls_optim, mit_load_data
from wls import filter_by_odom
from tools import poses_to_zero, integr
from scipy import sparse
import numpy as np
from scipy.sparse.linalg import lsqr
import scipy
import sys, os, logging

ALPHAS = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
          0.01, 0.025, 0.05, 0.075, 0.1, 0.5, 1.0]

def calc_p_deriv_i(i, s_phi, indexes):
    mask = (indexes[:, 0] <= i) & (i < indexes[:, 1])

    k = 0
    angles = []
    row_ind = []
    column_ind = []
    for m in np.nonzero(mask)[0]:
        v1 = indexes[m][1]
        angles.append(np.cumsum(s_phi[i:v1-1]))

        n = len(angles[-1])
        if n == 0:
            continue

        row_ind.append(m*np.ones(n))
        column_ind.append(np.arange(i+1, v1))

    angles = np.hstack(angles)
    row_ind = np.hstack(row_ind)
    column_ind = np.hstack(column_ind)

    shape = (len(indexes), len(s_phi))
    a_cos = sparse.coo_matrix((angles, (row_ind, column_ind)), shape)
    a_sin = a_cos.copy()
    a_cos.data = np.cos(a_cos.data)
    a_sin.data = np.sin(a_sin.data)

    a = sparse.hstack([-a_sin, -a_cos], format='coo')
    b = sparse.hstack([a_cos, -a_sin], format='coo')
    return sparse.vstack([a, b], format='coo')

def calc_p_deriv(s_xy, s_phi, w_xy, r_xy, e_xy, indexes):
    n = len(s_phi)
    p_deriv = np.zeros(n, np.float32)
    sparse_w_xy = sparse.diags(w_xy, 0, format='csr')
    for i in range(n-1):
        deriv = calc_p_deriv_i(i, s_phi, indexes)
        p_deriv[i] = (sparse_w_xy.dot(deriv).dot(s_xy)).T.dot(r_xy)
    return 1/e_xy*p_deriv


def comp_jacob(x, m, w_xy, A, P, alpha, indexes):
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

    j_phi += calc_p_deriv(s_xy, s_phi, w_xy, r_xy, e_xy, indexes)
    return np.hstack([j_xy, j_phi])


def f(s, m, w_xy, w_phi, alpha, indexes, indices, indptr):
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

def grad(s, m, w_xy, w_phi, alpha, indexes, indices, indptr):
    logging.debug('Jacobian start: %.3f' % alpha)
    l = len(s)//3
    s_xy = s[:2*l]
    s_phi = s[2*l:]

    P = build_a_xy(indexes, s_phi, indices, indptr)
    P = sparse.diags(w_xy, 0, format='csr').dot(P)

    A = sparse.csr_matrix((np.ones(len(indices), np.float32), indices, indptr))
    A = sparse.diags(w_phi, 0, format='csr').dot(A)

    j = comp_jacob(s, m, w_xy, A, P, alpha, indexes)
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

def nonlin_optim(s0, alpha, args):
    m, w_xy, w_phi, indexes, indices, indptr = args
    return scipy.optimize.minimize(f, s0, jac=grad, method='BFGS',
        args=(m, w_xy, w_phi, alpha, indexes, indices, indptr))

RHO = 10


def proc(dataset_n, start_index, end_index):
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

    logging.info('Linear optimization')
    d0 = wls_optim(match, cov, indexes, odom, perc=RHO)
    np.save('results/nonlinear/%d/%d_linear.npy' % (dataset_n, start_index), d0)

    d0 = np.hstack([d0[:, 0], d0[:, 1], d0[:, 2]])

    s0 = d0.copy()

    args = get_args(match, cov, indexes, odom, RHO)

    (m, w_xy, w_phi, indexes, indices, indptr) = args

    for alpha in reversed(ALPHAS):
        logging.info('==== Started %.3f', alpha)
        lin = f(d0, m, w_xy, w_phi, alpha, indexes, indices, indptr)
        init = f(s0, m, w_xy, w_phi, alpha, indexes, indices, indptr)

        res = nonlin_optim(s0, alpha, args)

        logging.info('Linear:\t%f', lin)
        logging.info('Init:\t%f', init)
        logging.info('After:\t%f', res.fun)

        l = len(res.x)//3
        res_data = np.array([res.x[:l], res.x[l:-l], res.x[-l:]]).T

        np.save('results/nonlinear/%d/%d_%.3f.npy' % (dataset_n, k, alpha),
            res_data)
        s0 = res.x


BLOCK = 600
STEP = 300

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO)

    dataset_n = int(sys.argv[1])
    gt = np.load('datasets/mit/ground_truth/%d.npy' % dataset_n)

    k = 0
    while k < len(gt) - BLOCK//2:
        logging.info('Started block: %d/%d', k, len(gt))
        start = k
        end = k+BLOCK

        proc(dataset_n, start, end)
        k += STEP
