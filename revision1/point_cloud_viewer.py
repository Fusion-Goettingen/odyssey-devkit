import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
import std_msgs.msg
import struct
import time

import odyssey_dataloader as dodyssey


class PointCloudPublisher(Node):

    def __init__(self):
        super().__init__('pointcloud_publisher')

        self.publisher_ = self.create_publisher(PointCloud2, 'point_cloud', 10)
        #self.timer = self.create_timer(1.0, self.publish_pointcloud)

    def create_pointcloud2(self, points: np.ndarray):
        """
        Convert Nx3 numpy array to PointCloud2
        """
        msg = PointCloud2()

        msg.header = std_msgs.msg.Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"

        msg.height = 1
        msg.width = points.shape[0]

        if points.shape[-1] == 3:
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            ]
        elif points.shape[-1] == 4:
            msg.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='t', offset=12, datatype=PointField.FLOAT32, count=1),
            ]

        msg.is_bigendian = False
        msg.point_step = 16  # 3 * 4 bytes
        msg.row_step = msg.point_step * points.shape[0]
        msg.is_dense = True

        # Convert numpy array to bytes
        buffer = []
        for p in points:
            buffer.append(struct.pack('ffff', p[0], p[1], p[2], p[3]))

        msg.data = b''.join(buffer)

        return msg

    def publish_pointcloud(self, points):
        msg = self.create_pointcloud2(points)
        self.publisher_.publish(msg)

        self.get_logger().info(f'Published {points.shape[0]} points')




def main(args=None):
    base_dir = "/media/aaron/OKULAr_HDD/odyssey_rev1"
    seq = "ForestRoad1"

    rclpy.init(args=args)
    node = PointCloudPublisher()
    # Showing the entire sequence

    """
    lidar_timestamps, lidar_poses = dodyssey.load_lidar_poses(base_dir,seq)
    idx = np.array(range(0,len(lidar_timestamps)))[::50]
    for id in idx:
        print(id)
        pc = dodyssey.load_pointcloud(base_dir,seq,lidar_timestamps[id])
        pc = pc[~np.isnan(pc[:,0])]
        points = np.ascontiguousarray(pc[:,:4])
        points[:,-1] = id / len(lidar_timestamps)
        points[:,:-1] = (lidar_poses[id,:3,:3] @ points[:,:-1].T).T + lidar_poses[id,:3,-1]
        node.publish_pointcloud(points)
    """

    """
    import matplotlib.pyplot as plt
    navsatfix = dodyssey.load_refsys_navsatfix(base_dir,seq)
    imu = dodyssey.load_refsys_imu(base_dir,seq)
    plt.plot(navsatfix[:,0], navsatfix[:,-3],label="Var lat")
    plt.plot(navsatfix[:,0], navsatfix[:,-2],label="Var lon")
    plt.plot(navsatfix[:,0], navsatfix[:,-1],label="Var alt")

    plt.plot(imu[:,0], imu[:,5], label="Var 00")
    plt.plot(imu[:,0], imu[:,6], label="Var 11")
    plt.plot(imu[:,0], imu[:,7], label="Var 22")
    plt.title("ForestRoad1: Varinances over time")
    plt.legend()
    plt.show()
    """
   
    """
    lidar_timestamps, lidar_poses = dodyssey.load_lidar_poses(base_dir,seq)
    frames = [0,len(lidar_timestamps)-1]
    for id in frames:
        pc = dodyssey.load_pointcloud(base_dir,seq,lidar_timestamps[id])
        pc = pc[~np.isnan(pc[:,0])]
        points = np.ascontiguousarray(pc[:,:4])
        points[:,-1] = id / len(lidar_timestamps)
        points[:,:-1] = (lidar_poses[id,:3,:3] @ points[:,:-1].T).T + lidar_poses[id,:3,-1]
        node.publish_pointcloud(points)
        time.sleep(0.5)

        lidar_timestamps, lidar_poses = dodyssey.load_lidar_poses(base_dir,seq)
    """

    lidar_timestamps, lidar_poses = dodyssey.load_lidar_poses(base_dir,seq)
    timestamps, poses = dodyssey.load_refsys_poses(base_dir,seq)
    frames = [0,len(lidar_timestamps)-1]

    pc = dodyssey.load_pointcloud(base_dir,seq,lidar_timestamps[0])
    pc = pc[~np.isnan(pc[:,0])]
    points = np.ascontiguousarray(pc[:,:4])
    points[:,-1] = 0
    points[:,:-1] = (poses[0,:3,:3] @ points[:,:-1].T).T + poses[0,:3,-1]
    #node.publish_pointcloud(points)

    time.sleep(0.5)

    pc = dodyssey.load_pointcloud(base_dir,seq,lidar_timestamps[-1])
    pc = pc[~np.isnan(pc[:,0])]
    points = np.ascontiguousarray(pc[:,:4])
    points[:,-1] = 1
    points[:,:-1] = (poses[-1,:3,:3] @ points[:,:-1].T).T + poses[-1,:3,-1]
    node.publish_pointcloud(points)
    

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()