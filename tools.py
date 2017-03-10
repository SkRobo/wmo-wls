import numpy as np

def integr(poses, init_pose=None):
    res = np.zeros((len(poses) + 1, 3), dtype=np.float32)
    if init_pose is not None:
        res[0] = init_pose
    for i, vec in enumerate(poses):
        res[i+1] = res[i]
        res[i+1][2] += vec[2]
        res[i+1][:2] += rot(vec[:2], res[i][2])
    return res

def integr_dist(poses):
    d = poses[1:] - poses[:-1]
    return np.sum(np.linalg.norm(d[:, :2], axis=1))


def rot(points, ang):
    c, s = np.cos(ang), np.sin(ang)
    if points.ndim == 1:
        return np.array([
            points[0]*c - points[1]*s,
            points[0]*s + points[1]*c,
        ], points.dtype)
    out = np.empty((len(points), 2), points.dtype)
    out[:, 0] = points[:, 0]*c - points[:, 1]*s
    out[:, 1] = points[:, 0]*s + points[:, 1]*c
    return out


def fix_poses_angl(poses):
    da = poses[1:, 2] - poses[:-1,2]
    da[abs(da) < 3] = 0
    da[da < -3] = -1
    da[da > 3] = 1
    nrot = np.cumsum(da)
    poses[1:, 2] -= 2*np.pi*nrot

def poses_to_zero(poses):
    fix_poses_angl(poses)
    res = poses.copy()
    res[:, :2] = rot(res[:, :2], -res[0, 2])
    res -= res[0].copy()
    return res


def c2xy(scan, fov=260, min_dist=0.02, dtype=np.float32):
    scan_size = len(scan)
    points = np.empty((scan_size, 2), dtype=dtype)
    angles = np.radians(np.linspace(-fov/2, fov/2, scan_size))
    points[:, 0] = scan*np.cos(angles)
    points[:, 1] = scan*np.sin(angles)
    return points[scan>min_dist]
