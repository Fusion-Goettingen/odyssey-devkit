import numpy as np
from pathlib import Path
from scipy.spatial.transform import Slerp, Rotation
import glob

dtype_navsatfix = np.dtype([("t", "i8"), ("lat", "f8"), ("lon", "f8"), ("alt", "f8"),("cov_00", "f8"), ("cov_11", "f8"), ("cov_22", "f8")]) # time, geodetic coodinates in ETRS89/ETRF2024 and diagonal of the covariance 

dtype_imu = np.dtype([
    ("t", "i8"),                                                                # time
    ("ori_x", "f8"), ("ori_y", "f8"), ("ori_z", "f8"), ("ori_w", "f8"),         # quatiernion x-y-z-w
    ("cov_ori_00", "f8"), ("cov_ori_11", "f8"), ("cov_ori_22", "f8"),           # diagonal of cov. of orientation
    ("angvel_x", "f8"), ("angvel_y", "f8"), ("angvel_z", "f8"),                 # angular velocity x-y-z
    ("cov_angvel_00", "f8"), ("cov_angvel_11", "f8"), ("cov_angvel_22", "f8"),  # diagonal of cov. of angular velocity
    ("linacc_x", "f8"), ("linacc_y", "f8"), ("linacc_z", "f8"),                 # linear acceleration x-y-z
    ("cov_linacc_00", "f8"), ("cov_linacc_11", "f8"), ("cov_linacc_22", "f8")]) # diagonal of cov. of linear acceleration

def load_data(base_dir, seq, folder, filename, dtype, as_structured=False):
    """
    Loads data from the specified location <base_dir/seq/folder/filename>
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    folder: folder
    filename: filename
    dtype: the dtype of the data
    as_structured: default false, return the data as a structured array instead of af a array of dtype float

    Returns
    -------
    the data stored in the file
    """
    file_path = Path(base_dir).joinpath(seq).joinpath(folder).joinpath(filename)
    data = np.genfromtxt(str(file_path), dtype=dtype)
    if as_structured:
        return data
    data = np.array(data.tolist(), dtype=float)
    return data

def load_pointcloud(base_dir, seq, timestamp, preserve_2D_order=False):
    """
    Loads the pointcloud corresponding to the given timestamp. Invlid points contain nan values for their coordinates
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    timestamp: timestamp of the point cloud
    preserve_2D_order: default False, preserves the 2D ordering as produced by the lidar

    Returns
    -------
    the data stored in the file
    """
    pc = np.fromfile(Path(base_dir).joinpath(seq).joinpath("ouster").joinpath(f"{timestamp}.bin"), np.float32).reshape((-1, 6))
    if preserve_2D_order:
        pc = pc.reshape((128, -1, pc.shape[-1]))
    return pc

def load_lidar_timestamps(base_dir,seq):
    timestamps = np.genfromtxt(str(Path(base_dir).joinpath(seq).joinpath("metadata").joinpath("lidar_timestamps.txt")),dtype=int)
    return timestamps

def pointcloud_generator(base_dir,seq,preserve_2D_order=False):
    timestamps = load_lidar_timestamps(base_dir,seq)
    timestamps.sort()

    for timestamp in timestamps:
        pc = load_pointcloud(base_dir,seq,timestamp,preserve_2D_order)
        yield timestamp, pc

def load_refsys_navsatfix(base_dir, seq, as_structured=False):
    return load_data(base_dir, seq, "refsys", "navsatfix.txt", dtype_navsatfix, as_structured)

def load_refsys_imu(base_dir, seq, as_structured=False):
    return load_data(base_dir, seq, "refsys", "imu.txt", dtype_imu, as_structured)


def load_m300_imu(base_dir, seq, as_structured=False):
    return load_data(base_dir, seq, "m300", "imu.txt", dtype_imu, as_structured)


def llas_to_cart(llas, origin=None):
    def lla_to_cart(lla, scale):
        lat = lla[0]
        lon = lla[1]
        alt = lla[2]
        er = 6378137.0
        tx = scale * lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + lat) * np.pi / 360.0))
        tz = alt
        t = np.array([tx, ty, tz])
        return t

    if origin is not None:
        llas = np.vstack((origin, llas))
        return llas_to_cart(llas)[1:]

    scale = None
    ts = np.zeros((len(llas), 3))

    for i, lla in enumerate(llas):
        lat = lla[0]
        if scale is None:
            scale = np.cos(lat * np.pi / 180.0)
        t = lla_to_cart(lla, scale)
        if origin is None:
            origin = t
        ts[i] = t - origin

    return ts


def closest_searchsorted(a, v):
    ii_right = np.searchsorted(a,v)
    ii_right = np.clip(ii_right,0,len(a)-1)
    ii_left = ii_right - 1
    ii_left = np.clip(ii_left,0,len(a)-1)
    dt_left = np.abs(v - a[ii_left])
    dt_right = np.abs(v - a[ii_right])
    ii = np.full(ii_left.shape,-1,int)
    ii[dt_left <= dt_right] = ii_left[dt_left <= dt_right]
    ii[dt_right < dt_left] = ii_right[dt_right < dt_left]
    import matplotlib.pyplot as plt
    return ii

def interpolate(key_times, key_value, times):
    if key_value.ndim == 1:
        return np.interp(times,key_times,key_value)
    elif key_value.ndim == 2:
        res = np.zeros((len(times), key_value.shape[1]),dtype=key_value.dtype)
        for i in range(key_value.shape[1]):
            res[:,i] = interpolate(key_times,key_value[:,i],times)
        return res
    else:
        raise ValueError("key_value with ndim > 2 not supported")


def combine_navsatfix_and_imu_at_times(timestamps, navsatfix, imu, match_mode="closest", to_cartesian=True, normalize_orientation=True):
    assert match_mode == "closest" or match_mode == "interpolate"

    navsatfix_timestamps = navsatfix[:, 0]
    navsatfix_data = navsatfix[:, 1:4]
    imu_timestamps = imu[:, 0]
    imu_data = imu[:, 1:]

    if match_mode == "closest":
        nav_idx = closest_searchsorted(navsatfix_timestamps, timestamps)
        imu_idx = closest_searchsorted(imu_timestamps, timestamps)
        positions = navsatfix_data[nav_idx]
        orientations = Rotation.from_quat(imu_data[imu_idx, :4],scalar_first=False)
    elif match_mode == "interpolate":
        positions = interpolate(navsatfix_timestamps, navsatfix_data, timestamps)
        slerp = Slerp(imu_timestamps, Rotation.from_quat(imu_data[:, :4],scalar_first=False))
        orientations = slerp(np.clip(timestamps, np.min(imu_timestamps), np.max(imu_timestamps)))

    if to_cartesian:
        positions = llas_to_cart(positions)

    if normalize_orientation:
        positions = orientations[0].inv().apply(positions)
        orientations = orientations[0].inv() * orientations

    poses = np.zeros((len(positions), 4, 4), dtype=float)
    poses[:, :3, -1] = positions
    poses[:, :3, :3] = orientations.as_matrix()
    poses[:, -1, -1] = 1
    return poses

def load_refsys_poses(base_dir, seq, timestamps=None, match_mode="closest", to_cartesian=True, normalize_orientation=True):
    navsatfix = load_refsys_navsatfix(base_dir, seq)
    imu = load_refsys_imu(base_dir, seq)
    if timestamps is None:
        gt_poses = combine_navsatfix_and_imu_at_times(imu[:,0], navsatfix, imu, match_mode, to_cartesian, normalize_orientation)
        return imu[:,0], gt_poses
    else:
        gt_poses = combine_navsatfix_and_imu_at_times(timestamps, navsatfix, imu, match_mode, to_cartesian, normalize_orientation)
        return timestamps, gt_poses

def load_ground_truth_poses(base_dir, seq):
    lidar_timestamps = load_lidar_timestamps(base_dir,seq)
    lidar_poses_3x4 = np.genfromtxt(Path(base_dir).joinpath(seq).joinpath("refsys").joinpath("ground_truth_poses.txt"),delimiter=" ").reshape((-1,3,4))
    lidar_poses = np.zeros((len(lidar_poses_3x4),4,4))
    lidar_poses[:,:3] = lidar_poses_3x4
    lidar_poses[:,-1,-1] = 1
    return lidar_timestamps, lidar_poses
