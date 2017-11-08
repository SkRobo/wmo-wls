#!/usr/bin/env python3
import numpy as np
import logging, io, os, subprocess, json, multiprocessing

SM2_PATH = './csm/sm2'
MIN_READING = 0.02
MAX_READING = 50

TRH = 1000

def get_lines(scans, odometry, indexes, fov):
    for scan, pose, index in zip(scans, odometry, indexes):
        data = {}
        theta = np.radians(fov/2)
        data['min_theta'] = -theta
        data['max_theta'] = theta
        data['theta'] = np.linspace(-theta, theta, len(scan)).tolist()
        data['nrays'] = len(scan)
        data['odometry'] = pose.tolist()
        mask = np.logical_and(scan>MIN_READING, scan<MAX_READING)
        data['valid'] = mask.astype(np.uint8).tolist()
        readings = scan.copy()
        readings[~mask] = 0
        data['readings'] = readings.tolist()
        data['timestamp'] = [int(index), 0]
        yield json.dumps(data)

def process(scans, odometry, indexes, fov, params={'do_compute_covariance': True}):
    if len(scans) > TRH:
        yield from process(scans[:TRH], odometry[:TRH], indexes[:TRH], fov, params)
        yield from process(scans[TRH-1:], odometry[TRH-1:], indexes[TRH-1:], fov, params)
        return

    cmd = [SM2_PATH]
    for key, value in params.items():
        if isinstance(value, bool):
            value = 1 if value else 0
        cmd += [key, str(value)]

    cmd += ['out_stats', 'stdout', 'out', '/dev/null']
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        universal_newlines=True, bufsize=1)

    inp_data = '\n'.join(get_lines(scans, odometry, indexes, fov))
    out_data = p.communicate(input=inp_data + '\n')[0]
    out = io.StringIO(out_data)

    for line in out.readlines():
        yield json.loads(line)

def proc(scans, odometry, step, start, fov):
    indexes = np.arange(len(scans))
    res = process(scans[start::step], odometry[start::step],
        indexes[start::step], fov)
    n = len(indexes[start::step])-1
    mask = np.zeros(n, np.bool)
    match = np.zeros((n, 3), np.float32)
    cov_x = np.zeros((n, 3, 3), np.float32)
    indexes = np.zeros((n, 2), np.uint32)
    iters = np.zeros(n, np.uint32)
    errs = np.zeros(n, np.float32)
    nvalid = np.zeros(n, np.uint16)
    i = 0
    for i, v in enumerate(res):
        mask[i] = v['valid'] == 1
        match[i] = v['x']
        cov_x[i] = v['cov_x']
        indexes[i] = [v['laser_ref_timestamp'][0], v['laser_sens_timestamp'][0]]
        iters[i] = v['iterations']
        errs[i] = v['error']
        nvalid[i] = v['nvalid']
    if i != n-1:
        logging.warning('Got %d/%d scan matcher failures' , n-1-i, n-1)
    logging.info('Done %d-%d', start, step)
    return match[mask], cov_x[mask], indexes[mask], iters[mask], errs[mask], nvalid[mask]

def run_pool(odom_path, scans_path, out_path, fov, win_size=20):
    odometry = np.load(odom_path)
    scans = np.load(scans_path)
    assert(len(odometry) == len(scans))

    pool = multiprocessing.Pool()
    tasks = []
    for step in range(1, win_size+1):
        for d in range(step):
            tasks.append((scans, odometry, step, d, fov))
    res = pool.starmap(proc, tasks)
    ind = np.vstack([v[2] for v in res])
    sort_ind = np.lexsort((ind[:, 0], ind[:, 1]))
    np.save(out_path % 'match', np.vstack([v[0] for v in res])[sort_ind])
    np.save(out_path % 'cov', np.vstack([v[1] for v in res])[sort_ind])
    np.save(out_path % 'indexes', ind[sort_ind])
    np.save(out_path % 'iters', np.hstack([v[3] for v in res])[sort_ind])
    np.save(out_path % 'errs', np.hstack([v[4] for v in res])[sort_ind])
    np.save(out_path % 'nvalid', np.hstack([v[5] for v in res])[sort_ind])
    pool.close()
    pool.join()





MIT_DATASETS = './datasets/mit/'
MIT_OUT = './results/match/mit/'
MIT_FOV = 260

SK_DATASETS = './datasets/skoltech/'
SK_OUT = './results/match/skoltech/'
SK_FOV = 240

def run_mit():
    logging.info('Started MIT')
    for v in ['match', 'cov', 'indexes', 'iters', 'errs', 'nvalid']:
        os.makedirs(MIT_OUT + v, exist_ok=True)

    for i in range(24):
        logging.info('Processing: %d' % i)
        scans_path = MIT_DATASETS  + 'scans/%d.npy' % i
        odom_path = MIT_DATASETS  + 'odometry/%d.npy' % i
        out_path = MIT_OUT + '%s/' + ('%d.npy' % i)
        run_pool(odom_path, scans_path, out_path, MIT_FOV)
    logging.info('Finished MIT')



def run_sk():
    logging.info('Started Skoltech')
    os.makedirs(SK_OUT, exist_ok=True)

    scans_path = SK_DATASETS  + 'scans.npy'
    odom_path = SK_DATASETS  + 'odometry.npy'
    out_path = SK_OUT + '%s.npy'
    run_pool(odom_path, scans_path, out_path, SK_FOV)
    logging.info('Finished Skoltech')


if __name__ == '__main__':
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        level=logging.INFO)
    run_sk()
    run_mit()
