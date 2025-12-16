
from odyssey_dataloader import *

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    base_dir = "/media/aaron/Odyssey/Odyssey"
    seq = "UndergroundCarPark1"

    """
    ****************************************
    *  SECTION: Loading Ground Truth Poses *
    ****************************************
    """
    # Loading the full list of ground truth poses, as cartesian and first pose set to identity
    timestamps, gt_poses_full = load_refsys_poses(base_dir,seq,True,True)
    plt.plot(gt_poses_full[:,0,-1],gt_poses_full[:,1,-1], c="C0")

    # Loading lidar poses from lidar_poses.txt file
    lidar_timestamps, lidar_poses = load_lidar_poses(base_dir,seq)
    plt.plot(lidar_poses[:,0,-1],lidar_poses[:,1,-1], c="C1")

    # Loading from file is just a convenience function. The same can be achieved using the following code
    lidar_timestamps = load_lidar_timestamps(base_dir,seq)
    lidar_timestamps_2, lidar_poses_2 = load_refsys_poses_at_times(base_dir,seq,lidar_timestamps,match_mode="closest",to_cartesian=True, normalize_orientation=True)

    assert np.all(lidar_timestamps == lidar_timestamps_2)
    assert np.all(np.isclose(lidar_poses,lidar_poses_2))

    plt.plot(lidar_poses_2[:,0,-1],lidar_poses_2[:,1,-1], c="C2")
    plt.show()

    """
    ***************************************
    *  SECTION: Loading Point Cloud Data  *
    ***************************************
    """

    # Loading the (first) pointcloud at time lidar_timestamps[0].
    lidar_timestamps = load_lidar_timestamps(base_dir, seq)
    pointcloud = load_pointcloud(base_dir,seq,lidar_timestamps[0],False)
    plt.scatter(pointcloud[:,0],pointcloud[:,1],s=2,c=pointcloud[:,4])
    plt.show()

    # Preserving the 2D structure of the pointcloud lets you interpret the lidar data as an image.
    lidar_timestamps = load_lidar_timestamps(base_dir, seq)
    pointcloud = load_pointcloud(base_dir,seq,lidar_timestamps[0], True)
    #range_image = np.linalg.norm(pointcloud[:,:,:3],axis=-1)
    plt.imshow(pointcloud[:,:,4])
    plt.show()

    # Pointcloud generator to iterate through all pointcloud of a sequence.
    pc_gen = pointcloud_generator(base_dir,seq)
    for timestamp, pointcloud in pc_gen:
        #plt.scatter(pointcloud[:,0],pointcloud[:,1],s=2,c=pointcloud[:,4])
        break

    """
    *******************************
    *  SECTION: Loading IMU data  *
    *******************************
    """

    # Loading imu data from the m300. Plotting the angular velocities and linear acceleration over time.
    imu_data = load_m300_imu(base_dir,seq,False)
    angvel = imu_data[:,8:11]
    linacc = imu_data[:,14:17]
    plt.plot(imu_data[:,0], angvel[:,0],c="C0",label="x")
    plt.plot(imu_data[:,0], angvel[:,1],c="C1",label="y")
    plt.plot(imu_data[:,0], angvel[:,2],c="C2",label="z")
    plt.legend()
    plt.show()

    plt.plot(imu_data[:,0], linacc[:,0],c="C0",label="x")
    plt.plot(imu_data[:,0], linacc[:,1],c="C1",label="y")
    plt.plot(imu_data[:,0], linacc[:,2],c="C2",label="z")
    plt.legend()
    plt.show()









