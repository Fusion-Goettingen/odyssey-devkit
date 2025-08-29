import os.path

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Slerp, Rotation
import glob

dtype_navsatfix = np.dtype([("t", "i8"), ("lat", "f8"), ("lon", "f8"), ("alt", "f8")])

dtype_imu = np.dtype([
    ("t", "i8"),  # time
    ("ox", "f8"), ("oy", "f8"), ("oz", "f8"), ("ow", "f8"),  # quatiernion x-y-z-w
    ("cov_o0", "f8"), ("cov_o1", "f8"), ("cov_o2", "f8"),  # diagonal of cov. of orientation
    ("avx", "f8"), ("avy", "f8"), ("avz", "f8"),  # angular velocity x-y-z
    ("cov_av0", "f8"), ("cov_av1", "f8"), ("cov_av2", "f8"),  # diagonal of cov. of angular velocity
    ("lax", "f8"), ("lay", "f8"), ("laz", "f8"),  # linear acceleration x-y-z
    ("cov_la0", "f8"), ("cov_la1", "f8"), ("cov_la2", "f8")])  # diagonal of cov. of linear acceleration

dtype_odometry = np.dtype(
    [("t", "i8"), ("x", "f8"), ("y", "f8"), ("z", "f8"), ("ox", "f8"), ("oy", "f8"), ("oz", "f8"), ("lvx", "f8"),
     ("lvy", "f8"), ("lvz", "f8"), ("avx", "f8"), ("avy", "f8"), ("avz", "f8")])

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
    preserve_beams: default False, preserves the 2D ordering as produced by the lidar

    Returns
    -------
    the data stored in the file
    """
    pc = np.fromfile(Path(base_dir).joinpath(seq).joinpath("ouster").joinpath(f"{timestamp}.bin"), np.float32).reshape(
        (-1, 6))
    if preserve_2D_order:
        pc = pc.reshape((128, -1, pc.shape[-1]))
    return pc


def load_seq_names(base_dir):
    """Given a base_dir, returns the folders inside in order to get the sequence names available to load"""
    subdirs = [entry.name for entry in os.scandir(base_dir) if entry.is_dir()]
    return subdirs

def load_refsys_navsatfix(base_dir, seq, as_structured=False):
    """
    Loads the NavSatFix messages of the reference system
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    as_structured: default false, return the data as a structured array instead of af a array of dtype float

    Returns
    -------
    the NavSatFix messages of the reference system
    """
    return load_data(base_dir, seq, "refsys", "navsatfix.txt", dtype_navsatfix, as_structured)

def load_refsys_imu(base_dir, seq, as_structured=False):
    """
    Loads the IMU messages of the reference system
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    as_structured: default false, return the data as a structured array instead of af a array of dtype float

    Returns
    -------
    the IMU messages of the reference system
    """
    return load_data(base_dir, seq, "refsys", "imu.txt", dtype_imu, as_structured)

def load_m300_imu(base_dir, seq, as_structured=False):
    """
    Loads the IMU messages of the M300
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    as_structured: default false, return the data as a structured array instead of af a array of dtype float

    Returns
    -------
    the IMU messages of the M300
    """
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


def interpolate(key_times, key_value, times):
    """
    Linearly interpolates between key_values at times key_times, similar to Slerp (but without the spherical)
    Parameters
    """
    N = len(key_times) - 1
    idx = np.searchsorted(key_times, times)

    l = idx - 1  # left neighbor
    r = idx  # right neighbor
    l[l < 0] = 0
    r[r >= key_times.shape[0]] = key_times.shape[0] - 1

    l_val = key_times[l]
    r_val = key_times[r]
    time_diff = (r_val - l_val)
    time_diff[time_diff == 0] = float("inf")
    t = (times - l_val) / time_diff

    return key_value[l] * (1 - t[:, np.newaxis]) + key_value[r] * t[:, np.newaxis]


def closest_searchsorted(a, v):
    """
    Searches for the index where a is closest to v (for all v if v is a list). Assumes that a is sorted in ascending order.
    Parameters
    ----------
    a: the array where v is searched for in.
    v: the value (or array) that is searched for

    Returns
    -------
    the index where a is closest to v
    """
    N = len(a) - 1
    idx = np.searchsorted(a, v, side="right")
    d_l = np.full(idx.shape[0], 10000000000000000)  # TODO: replace with proper int max value
    d_r = np.full(idx.shape[0], 10000000000000000)  # TODO: replace with proper int max value
    d_l[idx > 0] = np.abs(a[idx[idx > 0] - 1] - v[idx > 0])
    d_r[idx < N] = np.abs(a[idx[idx < N] + 1] - v[idx < N])
    d = np.zeros(idx.shape, int)
    d[d_l < d_r] = -1
    d[d_l > d_r] = 1
    id = idx + d
    return id


def combine_navsatfix_and_imu_at_times(timestamps, navsatfix, imu, match_mode="interpolate", to_cartesian=True,
                                       normalize_orientation=True):
    """
    Combines NavSatFix messages with corresponding IMU messages to 6DOF poses, depending on the match mode chosen.
    Parameters
    ----------
    timestamps: the timestamps at which the messages shall be combined
    navsatfix: the NavSatFix data
    imu: the IMU data
    match_mode: default "interpolate": interpolates the orientations spherically and the positions linearly to better fit timestamps, "closest": matches the closest NavSatFix and IMU messages to timestamp (for all elements in timestamps)
    to_cartesian: default True, returns the position in Cartesian coordinates (centered with the first position at the origin) instead of Geodetic coordinates
    normalize_orientation: default True, normalizes the orientation such that the first pose points towards the positive X axis.

    Returns
    -------
    the poses resulting from the positions of navsat and orientations from imu
    """
    assert match_mode == "closest" or match_mode == "interpolate"
    navsatfix_timestamps = navsatfix[:, 0]
    navsatfix_data = navsatfix[:, 1:]
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


def get_ground_truth_poses_timesynced_with_lidar(base_dir, seq, match_mode="interpolate", to_cartesian=True,
                                                 normalize_orientation=True):
    """
    Computes the ground truth poses from the sensor data of the reference system.
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    match_mode: default "interpolate": interpolates the orientations spherically and the positions linearly to better fit timestamps, "closest": matches the closest NavSatFix and IMU messages to timestamp (for all elements in timestamps)
    to_cartesian: default True, returns the position in Cartesian coordinates (centered with the first position at the origin) instead of Geodetic coordinates
    normalize_orientation: default True, normalizes the orientation such that the first pose points towards the positive X axis.

    """
    lidar_timestamps = load_data(base_dir, seq, "metadata", "lidar_timestamps.txt", float)
    navsatfix = load_refsys_navsatfix(base_dir, seq)
    imu = load_refsys_imu(base_dir, seq)
    gt_poses = combine_navsatfix_and_imu_at_times(lidar_timestamps, navsatfix, imu, match_mode, to_cartesian,
                                                  normalize_orientation)
    return gt_poses


def lidar_timestamps(base_dir, seq):
    """
    Returns an ordered list containing all timestamps of the lidar scans.
    Parameters
    ----------
    base_dir: base dir of the dataset
    seq: sequence name
    """
    lidar_files = glob.glob(str(Path(base_dir).joinpath(seq).joinpath("ouster").joinpath("*")))
    lidar_files = [file.replace(".bin", "").split("/")[-1] for file in lidar_files]
    lidar_timestamps = np.array(lidar_files, int)
    lidar_timestamps.sort()
    return lidar_timestamps

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    base_dir = "/media/aaron/OKULAr_HDD/odyssey_ecmr/"
    seq = "Feldweg1"

    # Loads ground truth poses, timesynced with lidar frames
    gt_poses = get_ground_truth_poses_timesynced_with_lidar(base_dir, seq, "interpolate", True, True)
    plt.plot(gt_poses[:, 0, -1], gt_poses[:, 1, -1], label="refsys")
    plt.axis("equal")
    plt.show()

    l_timestamps = lidar_timestamps(base_dir,seq)

    point_cloud = load_pointcloud(base_dir,seq,l_timestamps[0],False)

    plt.scatter(point_cloud[:,0],point_cloud[:,1],c="C0",s=1)
    plt.axis("equal")
    plt.show()

    # Loads the IMU data from the M300 system
    imu_data = load_data(base_dir, seq, "m300", "imu.txt", dtype_imu)
    imu_timestamp = imu_data[:, 0]
    imu_data = imu_data[:, 1:]