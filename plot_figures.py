#!/usr/bin/env python3
from plotter import plt, plot_grey_map, plot_diff_abs_comp
from tools import poses_to_zero, integr_dist, rot, c2xy, integr
from wls import mit_load_data, build_a_xy, build_indices
from scipy import sparse
from scipy.sparse.linalg import lsqr
from matplotlib import rc
from nonlinear import ALPHAS
import numpy as np
import os

os.makedirs('./figures/mit/', exist_ok=True)
os.makedirs('./figures/skoltech/', exist_ok=True)

# ================= check sk relation ==============

def sk_err(method, poses):
    relation = np.array([-0.46819474, -0.06316625,  0.3204352 ])
    delta = relation - poses[-1]
    xy = np.linalg.norm(delta[:2])
    angle = delta[2] - 2*np.pi
    print('%s\t%f\t%f\t%f\t%f' % (
        method, xy, xy/140*1000, angle, np.degrees(angle)))


def sk_calc_error(i = 20):
    print('\t\tXY, m\t\tXY, mm/m\tAngle, rad\tAngle, deg')
    sk_err('Frame-to-frame', np.load('results/wls/skoltech/%d.npy' % 1))
    sk_err('Keyframe', np.load('results/keyframe/skoltech/kf.npy'))
    sk_err('WMO-WLS\t', np.load('results/wls/skoltech/%d.npy' % i))

# ================= keyframes ======================
def sk_plot_cloud(name, poses):
    scans = np.load('datasets/skoltech/scans.npy')
    plot_grey_map(scans, poses, 240, ms=0.5)
    plt.tight_layout()
    plt.savefig('./figures/skoltech/lab_%s.png' % name, dpi=300)

def sk_plot_all_clouds():
    sk_plot_cloud('kf', np.load('results/keyframe/skoltech/kf.npy'))
    sk_plot_cloud('opt', np.load('results/wls/skoltech/%d.npy' % 20))
    sk_plot_cloud('simp', np.load('results/wls/skoltech/%d.npy' % 1))


# ================ MIT RMSE ===================

BLOCK = 1200
STEP = 600

def calc_rms_kf():
    data = []
    dists = []
    for n in range(24):
        kf = np.load('./results/keyframe/mit/%d.npy' % n)
        gt = poses_to_zero(np.load('./datasets/mit/ground_truth/%d.npy' % n))
        #
        k = 0
        while k < len(gt) - BLOCK//2:
            kf_chunk = poses_to_zero(kf[k:k+BLOCK])
            gt_chunk = poses_to_zero(gt[k:k+BLOCK])
            #
            d = integr_dist(gt_chunk)
            dists.append(d)
            diff = np.linalg.norm(kf_chunk[-1, :2] - gt_chunk[-1, :2])
            data.append(diff/d)
            #
            k += STEP
    data = np.array(data)
    return np.sqrt(np.mean(data**2))

def calc_rms(win):
    data = []
    dists = []
    for n in range(24):
        optim = np.load('./results/wls/mit/%d/%d.npy' % (win, n))
        gt = poses_to_zero(np.load('./datasets/mit/ground_truth/%d.npy' % n))
        #
        k = 0
        while k < len(gt) - BLOCK//2:
            optim_chunk = poses_to_zero(optim[k:k+BLOCK])
            gt_chunk = poses_to_zero(gt[k:k+BLOCK])
            #
            d = integr_dist(gt_chunk)
            dists.append(d)
            diff = np.linalg.norm(optim_chunk[-1, :2] - gt_chunk[-1, :2])
            data.append(diff/d)
            #
            k += STEP
    data = np.array(data)
    return np.sqrt(np.mean(data**2))

def mit_rmse():
    plt.figure()
    data = [calc_rms(i) for i in range(1, 21)]
    plt.plot(range(1, 21), data, color='black')
    plt.plot(range(1, 21), [calc_rms_kf()]*20, '--', color='black')
    plt.grid()
    plt.xlabel('Window size')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig('./figures/mit/rmse.eps', dpi=300)


# ================= MIT comparison ==================
def mit_comparison(n=10):
    plt.figure()
    gt = poses_to_zero(np.load('./datasets/mit/ground_truth/%d.npy' % n))
    poses = np.load('./results/wls/mit/20/%d.npy' % n)
    kf = np.load('./results/keyframe/mit/%d.npy' % n)

    plot_diff_abs_comp(gt, poses, kf)
    plt.tight_layout()
    plt.savefig('./figures/mit/result_%d.png' % n, dpi=300)

# ================= MIT Point cloud ==================
def mit_cloud():
    plt.figure(figsize=(10, 5))
    scans = np.load('datasets/mit/scans/0.npy')
    poses = poses_to_zero(np.load('datasets/mit/ground_truth/0.npy'))
    n = len(scans)
    for i, scan, v in zip(range(n), scans[::40], poses[::40]):
        p = rot(c2xy(scan, 260), v[2]) + v[:2]
        plt.plot(p[:, 0], p[:, 1], '.', color='black', ms=0.5)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.tight_layout()
    plt.savefig('./figures/mit/point_cloud.png', dpi=300)

# ================= Nonlinear ==================

def calc_xy_err(s, m, w, indexes):
    l = len(s)//3
    s_xy = s[:2*l]
    s_phi = s[2*l:]
    l2 = len(m)//3
    m_xy = m[:2*l2]
    w_xy = w[:2*l2]
    indices, indptr = build_indices(indexes)
    P = build_a_xy(indexes, s_phi, indices, indptr)
    P = sparse.diags(w_xy, 0, format='csr').dot(P)
    r_xy = (P.dot(s_xy) - m_xy*w_xy).astype(np.float64)
    return np.linalg.norm(r_xy)

def calc_phi_err(s, m, w, indexes):
    l = len(s)//3
    s_xy = s[:2*l]
    s_phi = s[2*l:]
    l2 = len(m)//3
    l2 = len(m)//3
    m_xy = m[:2*l2]
    m_phi = m[2*l2:]
    w_phi = w[2*l2:]
    indices, indptr = build_indices(indexes)
    A = sparse.csr_matrix((np.ones(len(indices), np.float64), indices, indptr))
    A = sparse.diags(w_phi, 0, format='csr').dot(A)
    r_phi = (A.dot(s_phi) - m_phi*w_phi).astype(np.float64)
    return np.linalg.norm(r_phi)

# TODO replace with linear.npy
def lin_opt(match, cov, indexes):
    indices, indptr = build_indices(indexes)

    # angle subsystem optimization
    w_phi = 1/cov[:, 2, 2]
    A = sparse.csr_matrix((np.ones(len(indices), np.float64), indices, indptr))
    A = sparse.diags(w_phi, 0, format='csr').dot(A)
    m_phi = match[:, 2]*w_phi
    phi_opt = lsqr(A, m_phi)[0]

    w_xy = 1/np.hstack([cov[:, 0, 0], cov[:, 1, 1]])
    P = build_a_xy(indexes, phi_opt, indices, indptr)
    P = sparse.diags(w_xy, 0, format='csr').dot(P)
    m_xy = np.hstack([match[:, 0], match[:, 1]])*w_xy

    xy_opt = lsqr(P, m_xy)[0]

    return np.hstack([xy_opt, phi_opt])

def calc_dist_diff(gt, s):
    l = len(s)//3
    v = integr(np.vstack([s[:l], s[l:-l], s[-l:]]).T)
    return np.linalg.norm(v[-1, :2] - gt[-1, :2])

def mit_nonlin_errs_comp(dataset_n=1):
    path = 'results/nonlinear/%d/' % dataset_n
    data = [np.load(path + '%.3f.npy' % v) for v in ALPHAS]
    match, cov, indexes, odom = mit_load_data(dataset_n)
    gt = poses_to_zero(np.load('datasets/mit/ground_truth/%d.npy' % dataset_n))

    m = np.hstack([
        match[:, 0],
        match[:, 1],
        match[:, 2],
    ])
    w = np.hstack([
        1/cov[:, 0, 0],
        1/cov[:, 1, 1],
        1/cov[:, 2, 2],
    ])

    s0 = lin_opt(match, cov, indexes)
    xy_err0 = calc_xy_err(s0, m, w, indexes)
    phi_err0 = calc_phi_err(s0, m, w, indexes)


    xy_err = np.array([calc_xy_err(s, m, w, indexes) for s in data])
    phi_err = np.array([calc_phi_err(s, m, w, indexes) for s in data])

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(ALPHAS, xy_err/xy_err0, color='black', label=r'$\varepsilon^{xy} (\alpha)/\varepsilon^{xy} (+ \infty)$')
    ax.plot(ALPHAS, phi_err/phi_err0, '--', color='black', label=r'$\varepsilon^a (\alpha)/\varepsilon^a (+ \infty)$')
    plt.legend()
    plt.ylim([0.9, 1.2])
    plt.xlim([0.001, 1])
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Ratio')
    ax.set_xscale("log", nonposx='clip')
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures/mit/nonlin_err_%d.eps' % dataset_n, dpi=300)

    plt.figure()
    ax = plt.subplot(111)
    d2_0 = calc_dist_diff(gt, s0)
    d2 = [calc_dist_diff(gt, s) for s in data]
    ax.plot(ALPHAS, d2/d2_0, 'black')
    plt.xlim([0.001, 1])
    plt.ylim([0., 1.5])
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Ratio')
    ax.set_xscale("log", nonposx='clip')
    plt.grid()
    plt.savefig('figures/mit/nonlin_dist_%d.eps' % dataset_n, dpi=300)

if __name__ == '__main__':
    rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    rc('text', usetex=True)

    print('sk_calc_error')
    sk_calc_error()
    print('sk_plot_all_clouds')
    sk_plot_all_clouds()
    print('mit_rmse')
    mit_rmse()
    print('mit_comparison')
    mit_comparison()
    print('mit_cloud')
    mit_cloud()
    '''
    mit_nonlin_errs_comp(1)
    mit_nonlin_errs_comp(2)
    mit_nonlin_errs_comp(3)
    mit_nonlin_errs_comp(10)
    '''