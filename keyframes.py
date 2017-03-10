#!/usr/bin/env python3
from tools import rot
import numpy as np
from csm import CSM
import progressbar, logging, os

MIT_DATASET = './datasets/mit/'
MIT_OUT = './results/keyframe/mit/'
MIT_FOV = 260

SK_DATASET = './datasets/skoltech/'
SK_OUT = './results/keyframe/skoltech/'
SK_FOV = 240

def keyframe_integr(scans, odom, fov, csm_params=None, kf_xy=0.1, kf_ang=0.175):
    csm = CSM(scan_size=scans.shape[1], scan_fov=fov, params=csm_params)
    poses = np.zeros((len(scans), 3), np.float32)
    keyframe = 0
    bar = progressbar.ProgressBar()
    for i in bar(range(1, len(scans))):
        pred = odom[i] - odom[keyframe]
        pred[:2] = rot(pred[:2], -odom[keyframe, 2])
        res = csm.match(scans[keyframe], scans[i], pred)
        if not res.is_valid:
            logging.warning('CSM failed: %d -> %d', keyframe, i)
            keyframe = i - 1
            res_x = pred
        else:
            res_x = res.x
        poses[i] = poses[keyframe]
        poses[i, 2] += res_x[2]
        poses[i][:2] += rot(res.x[:2], poses[keyframe, 2])
        if np.linalg.norm(res.x[:2]) > kf_xy or res.x[2] > kf_ang:
            keyframe = i
    return poses

def run_mit():
    logging.info('Started MIT')
    os.makedirs(MIT_OUT, exist_ok=True)
    for i in range(24):
        logging.info('Processing: %d' % i)
        scans = np.load(MIT_DATASET + 'scans/%d.npy' % i)
        odom = np.load(MIT_DATASET + 'odometry/%d.npy' % i)
        poses = keyframe_integr(scans, odom, MIT_FOV)
        np.save(MIT_OUT + '%d.npy' % i, poses)
    logging.info('Finished MIT')

def run_sk():
    logging.info('Started Skoltech')

    os.makedirs(SK_OUT, exist_ok=True)

    scans = np.load(SK_DATASET + 'scans.npy')
    odom = np.load(SK_DATASET + 'odometry.npy')
    poses = keyframe_integr(scans, odom, SK_FOV)
    np.save(SK_OUT + 'kf.npy', poses)

    logging.info('Finished Skoltech')

if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO)
    run_sk()
    run_mit()
