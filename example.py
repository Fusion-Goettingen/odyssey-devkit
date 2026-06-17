
from odyssey_dataloader import *

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    base_dir = "/path/to/odyssey_base_dir/"
    seq = "ParkingGarage1"

    """
    ****************************************
    *  SECTION: Loading Ground Truth Poses *
    ****************************************
    """

    # Loading ground truth poses from refsys/ground_truth_poses.txt (10Hz). Use this for evaluation.
    # This data is derived from the reference system as described in the paper.
    timestamps, gt_poses = load_ground_truth_poses(base_dir,seq)
    plt.plot(gt_poses[:,0,-1],gt_poses[:,1,-1], c="C1")
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
    # In this case we are showing the reflectivity.
    lidar_timestamps = load_lidar_timestamps(base_dir, seq)
    pointcloud = load_pointcloud(base_dir,seq,lidar_timestamps[0], True)
    #range_image = np.linalg.norm(pointcloud[:,:,:3],axis=-1)
    plt.imshow(pointcloud[:,:,4])
    plt.show()

    # You can also use a pointcloud generator to iterate through all pointcloud of a sequence.
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