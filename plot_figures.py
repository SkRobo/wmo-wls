#!/usr/bin/env python3
from plotter import plt, plot_grey_map, plot_diff_abs_comp
from tools import poses_to_zero, integr_dist, rot, c2xy, integr
from wls import mit_load_data, build_a_xy, build_indices
from scipy import sparse
import progressbar, glob, multiprocessing
from scipy.sparse.linalg import lsqr
from matplotlib import rc
from nonlinear import ALPHAS
import numpy as np
import os

# ================= pose graphs ====================

def plot_adj_f2f():
    plt.figure(figsize=(2, 2))
    a = np.zeros((20, 20), np.bool)
    a[:-1, 1:] = np.eye(19)
    plt.spy(a)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./figures/adj_f2f.eps', dpi=300)

def plot_adj_kf():
    plt.figure(figsize=(2, 2))
    a = np.zeros((20, 20), np.bool)
    a[0, 1:4] = 1
    a[3, 4:10] = 1
    a[9, 10:15] = 1
    a[14, 15:18] = 1
    a[17, 18:] = 1
    plt.spy(a)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./figures/adj_kf.eps', dpi=300)

def plot_adj_opt():
    plt.figure(figsize=(2, 2))
    a = np.zeros((20, 20), np.bool)
    for i in range(1, 6):
        a[:-i, i:] |= np.eye(20-i, dtype=np.bool)
    plt.spy(a)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./figures/adj_opt.eps', dpi=300)

# ================= check sk relation ==============

def sk_err(method, poses):
    relation = np.array([-0.46819474, -0.06316625,  0.3204352 ])
    delta = relation - poses[-1]
    xy = np.linalg.norm(delta[:2])
    angle = delta[2] - 2*np.pi
    print('%s\t%f\t%f\t%f\t%f' % (
        method, xy, xy/140*1000, angle, np.degrees(angle)))

def sk_calc_error(win=10):
    print('\t\tXY, m\t\tXY, mm/m\tAngle, rad\tAngle, deg')
    sk_err('Frame-to-frame', np.load('results/wls/skoltech/%d.npy' % 1))
    sk_err('Keyframe', np.load('results/keyframe/skoltech/kf.npy'))
    sk_err('WMO-WLS\t', np.load('results/wls/skoltech/%d.npy' % win))

# ================= keyframes ======================
def sk_plot_cloud(name, poses, ax2_lim):
    scans = np.load('datasets/skoltech/scans.npy')
    plot_grey_map(scans[::10], poses[::10], 240, ms=0.5, ax2_lim=ax2_lim)
    plt.tight_layout()
    plt.savefig('./figures/skoltech/lab_%s.png' % name, dpi=300)

def sk_plot_all_clouds(win=10):
    sk_plot_cloud('kf', np.load('results/keyframe/skoltech/kf.npy'),
        [-14, -6, -38, -32])
    sk_plot_cloud('opt', np.load('results/wls/skoltech/%d.npy' % win),
        [-13, -5, -39, -33])
    sk_plot_cloud('f2f', np.load('results/wls/skoltech/%d.npy' % 1),
        [-13, -5, -37, -31])


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
    plt.plot(range(2, 21), data[1:], color='black', label='WMO-WLS')
    plt.plot(range(1, 21), data[:1]*20, '-.',
        color='black', label='Frame-to-frame')
    plt.plot(range(1, 21), [calc_rms_kf()]*20, '--',
        color='black', label='Keyframe')
    plt.legend()
    plt.xlim([2, 20])
    plt.grid()
    plt.xlabel('Window size')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.savefig('./figures/mit/rmse.eps', dpi=300)


# ================= MIT comparison ==================
def mit_comparison(n=10, end=None):
    plt.figure()
    gt = poses_to_zero(np.load('./datasets/mit/ground_truth/%d.npy' % n))
    poses = np.load('./results/wls/mit/20/%d.npy' % n)
    kf = np.load('./results/keyframe/mit/%d.npy' % n)

    plot_diff_abs_comp(gt[:end], poses[:end], kf[:end])
    plt.tight_layout()
    plt.savefig('./figures/mit/result_%d.png' % n, dpi=300)

# ================= MIT Point cloud ==================
def mit_cloud():
    plt.figure(figsize=(8, 4))
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

def filter_params(path):
    part1, part2 = path.split('/')[-2:]
    dataset_n = int(part1)
    part3, part4 = part2.replace('.npy', '').split('_')
    start = int(part3)
    alpha = float(part4) if part4 != 'linear' else None
    return dataset_n, start, alpha

def filtered_data(dataset_n, start_index, end_index):
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
    m = np.hstack([match[:, 0], match[:, 1], match[:, 2]])
    w = np.hstack([1/cov[:, 0, 0]**2, 1/cov[:, 1, 1]**2, 1/cov[:, 2, 2]**2])
    return m, w, indexes

def cacl_errs(dataset_n, start, alpha):
    if alpha is not None:
        path = 'results/nonlinear/%d/%d_%.3f.npy' % (dataset_n, start, alpha)
    else:
        path = 'results/nonlinear/%d/%d_linear.npy' % (dataset_n, start)
    opt = np.load(path)
    m, w, indexes = filtered_data(dataset_n, start, start + BLOCK)
    s = np.hstack([opt[:, 0], opt[:, 1], opt[:, 2]])
    return calc_xy_err(s, m, w, indexes), calc_phi_err(s, m, w, indexes)

def mit_nonlin_err(dataset_n=10, start=0):
    path_pattern = 'results/nonlinear/%d/%d_*.npy' % (dataset_n, start)
    paths = glob.glob(path_pattern)
    params = [filter_params(path) for path in paths]

    alphas = sorted(set([
        alpha for _, _, alpha in params
        if alpha is not None
    ]))

    xy_errs = []
    phi_errs = []

    for alpha in alphas:
        xy_err, phi_err = cacl_errs(dataset_n, start, alpha)
        xy_errs.append(xy_err)
        phi_errs.append(phi_err)

    lin_xy, lin_phi = cacl_errs(dataset_n, start, None)
    xy_errs = np.array(xy_errs)/lin_xy
    phi_errs = np.array(phi_errs)/lin_phi

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(alphas, xy_errs, color='black',
        label=r'$\varepsilon^{xy} (\alpha)/\varepsilon^{xy} (+ \infty)$')
    ax.plot(alphas, phi_errs, '--', color='black',
        label=r'$\varepsilon^a (\alpha)/\varepsilon^a (+ \infty)$')
    plt.legend()
    plt.ylim([0.9, 1.2])
    plt.xlim([0.001, 1])
    plt.xlabel(r'$\alpha$')
    ax.set_xscale("log", nonposx='clip')
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures/mit/nonlin_err.eps', dpi=300)

def mit_nonlin_dist():
    paths = glob.glob('results/nonlinear/*/*.npy')
    params = [filter_params(path) for path in paths]

    alphas = sorted(set([alpha for _, _, alpha in params if alpha is not None]))
    results = dict((alpha, []) for alpha in alphas)
    results[None] = []
    bar = progressbar.ProgressBar(max_value=len(paths))
    for path, (dataset_n, start, alpha) in bar(zip(paths, params)):
        opt = integr(np.load(path))
        # bug in the nonlinear.py line 144
        if len(opt) == BLOCK + 1:
            opt = opt[:1200]
        gt_path = './datasets/mit/ground_truth/%d.npy' % dataset_n
        gt = np.load(gt_path)
        gt = poses_to_zero(gt[start:start+BLOCK])
        assert(len(opt) == len(gt))
        res = np.linalg.norm(opt[-1, :2] - gt[-1, :2])/integr_dist(gt)
        results[alpha].append(res)
    res = np.array([
        np.sqrt(np.mean(np.array(results[alpha])**2))
        for alpha in alphas
    ])
    lin_res = np.sqrt(np.mean(np.array(results[None])**2))
    res /= lin_res

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(alphas, res, 'black', label=r'$RMSE(\alpha)/RMSE(+\infty)$')
    plt.legend()
    plt.xlim([0.001, 1])
    plt.ylim([0.95, 1.05])
    plt.xlabel(r'$\alpha$')
    ax.set_xscale("log", nonposx='clip')
    plt.grid()
    plt.savefig('figures/mit/nonlin_dist.eps', dpi=300)

if __name__ == '__main__':
    os.makedirs('./figures/mit/', exist_ok=True)
    os.makedirs('./figures/skoltech/', exist_ok=True)

    font = {'family':'sans-serif','sans-serif':['Helvetica'],
            'size': 16}
    rc('font',**font)
    rc('text', usetex=True)

    print('misc')
    plot_adj_f2f()
    plot_adj_kf()
    plot_adj_opt()

    print('sk_calc_error')
    sk_calc_error(10)

    print('sk_plot_all_clouds')
    sk_plot_all_clouds(10)

    print('mit_rmse')
    mit_rmse()
    print('mit_comparison')
    mit_comparison(end=3500)
    print('mit_cloud')
    mit_cloud()

    print('nonlin dist err')
    mit_nonlin_dist()

    print('nonlin err')
    mit_nonlin_err()
