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

def load_lidar_timestamps(base_dir, seq):
    lidar_files = glob.glob(str(Path(base_dir).joinpath(seq).joinpath("ouster").joinpath("*")))
    lidar_files = [file.split("/")[-1].replace(".bin", "") for file in lidar_files]
    lidar_timestamps = np.array(lidar_files, int)
    lidar_timestamps.sort()
    return lidar_timestamps

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



def combine_navsatfix_and_imu_at_times(timestamps, navsatfix, imu, match_mode="closest", to_cartesian=True,
                                       normalize_orientation=True):
    assert match_mode == "closest" or match_mode == "interpolate"

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
    
    def interpolate(key_times, key_value, times):
        N = len(key_times) - 1
        idx = np.searchsorted(key_times, times)

        l = idx - 1 
        r = idx 
        l[l < 0] = 0
        r[r >= key_times.shape[0]] = key_times.shape[0] - 1

        l_val = key_times[l]
        r_val = key_times[r]
        time_diff = (r_val - l_val)
        time_diff[time_diff == 0] = float("inf")
        t = (times - l_val) / time_diff

        return key_value[l] * (1 - t[:, np.newaxis]) + key_value[r] * t[:, np.newaxis]



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


def load_refsys_poses(base_dir, seq, to_cartesian=True, normalize_orientation=True):
    navsatfix = load_refsys_navsatfix(base_dir, seq)
    imu = load_refsys_imu(base_dir, seq)
    gt_poses = combine_navsatfix_and_imu_at_times(imu[:,0], navsatfix, imu, "closest", to_cartesian,
                                                    normalize_orientation)
    return imu[:,0], gt_poses

def load_refsys_poses_at_times(base_dir, seq, timestamps, match_mode="closest", to_cartesian=True, normalize_orientation=True):
    navsatfix = load_refsys_navsatfix(base_dir, seq)
    imu = load_refsys_imu(base_dir, seq)
    gt_poses = combine_navsatfix_and_imu_at_times(timestamps, navsatfix, imu, match_mode, to_cartesian, normalize_orientation)
    return timestamps, gt_poses

def load_lidar_poses(base_dir, seq):
    lidar_timestamps = load_lidar_timestamps(base_dir,seq)
    poses_3x4 = np.genfromtxt(Path(base_dir).joinpath(seq).joinpath("refsys").joinpath("lidar_poses.txt"),dtype=float,delimiter=" ").reshape((-1,3,4))
    poses = np.zeros((len(poses_3x4),4,4),dtype=float)
    poses[:,:3] = poses_3x4
    poses[:,-1,-1] = 1

    return lidar_timestamps, poses
