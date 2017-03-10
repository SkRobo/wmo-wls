import matplotlib.pyplot as plt
from tools import rot, c2xy
import numpy as np

def plot_grey_map(scans, poses, fov=240, cmap=plt.cm.winter,
             poses_label=None, ms=None, color='grey'):
    plt.figure()
    n = len(scans)
    for i, scan, v in zip(range(n), scans, poses):
        p = rot(c2xy(scan, fov), v[2]) + v[:2]
        plt.plot(p[:, 0], p[:, 1], '.', color=color, ms=ms)
    plt.plot(poses[:, 0], poses[:, 1], 'black')
    plt.axis('equal')
    plt.xlabel('X, m')
    plt.ylabel('Y, m')
    plt.grid()
    plt.tight_layout()

def plot_diff_abs_comp(gt_poses, opt, kf, labels=None):
    plt.figure()
    ax1 = plt.subplot(211)

    diff = np.linalg.norm(kf[:, :2] - gt_poses[:, :2], axis=1,)
    plt.plot(diff, label='Keyframe', color='#999999')

    diff = np.linalg.norm(opt[:, :2] - gt_poses[:, :2], axis=1)
    plt.plot(diff, label='WMO-WLS', color='black')

    plt.legend(loc=2)
    plt.ylabel('Position difference, m')
    plt.grid()

    ax2 = plt.subplot(212, sharex=ax1)

    plt.plot(np.abs(kf[:, 2] - gt_poses[:, 2]), color='#999999')
    plt.plot(np.abs(opt[:, 2] - gt_poses[:, 2]), color='black')

    plt.ylabel('Angle difference, rad')
    plt.xlabel('Scan number')
    plt.grid()
