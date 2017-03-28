import matplotlib.pyplot as plt
from tools import rot, c2xy
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

def plot_grey_map(scans, poses, fov=240, cmap=plt.cm.winter,
             poses_label=None, ms=None, color='grey',
             ax2_lim=[-13, -5, -39, -33]):
    plt.figure()
    fig, ax = plt.subplots()

    axins = zoomed_inset_axes(ax, 3, loc=1)
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    axins2 = zoomed_inset_axes(ax, 3, loc=4)
    plt.xticks(visible=False)
    plt.yticks(visible=False)


    n = len(scans)
    for i, scan, v in zip(range(n), scans, poses):
        p = rot(c2xy(scan, fov), v[2]) + v[:2]
        ax.plot(p[:, 0], p[:, 1], '.', color=color, ms=ms)
        axins.plot(p[:, 0], p[:, 1], '.', color=color, ms=ms)
        axins2.plot(p[:, 0], p[:, 1], '.', color=color, ms=ms)
    ax.plot([], ':', color=color, label='Scans')
    ax.plot(poses[:, 0], poses[:, 1], 'black', label='Trajectory')
    axins.plot(poses[:, 0], poses[:, 1], 'black')
    axins2.plot(poses[:, 0], poses[:, 1], 'black')


    axins.axis('equal')
    axins.set_xlim(-3, 5)
    axins.set_ylim(-4, 2)

    axins2.axis('equal')
    axins2.set_xlim(ax2_lim[0], ax2_lim[1])
    axins2.set_ylim(ax2_lim[2], ax2_lim[3])

    ax.axis('equal')
    ax.set_xlim(-35, 40)
    ax.set_xlabel('X, m')
    ax.set_ylabel('Y, m')
    ax.grid()

    ax.legend(loc=7)
    mark_inset(ax, axins, loc1=2, loc2=3, ls='--', fc="none", ec="0.2")
    mark_inset(ax, axins2, loc1=2, loc2=3, ls='--', fc="none", ec="0.2")
    #plt.tight_layout()

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
