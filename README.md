# Odyssey-Dev-Kit

This is the Odyssey Dev-Kit containing some quality-of-life functions for loading our data set.
We provide functionalities for loading poses, IMU data, and lidar point clouds.
Please note, the dataset is not fully released, and the functionality of the software might change in the future.
If you find any quality-of-life functionalities to be missing, do not hesitate and contact us.

# Python Dependencies
- Numpy
- Scipy
- Matplotlib

# Quickstart
Go into the script `odyssey_dataloader.py` and change the `home_dir` to the path of your Odyssey data set. When executing the script, a window with the ground truth trajectory should open. After closing the window, a scatter plot of a lidar scan will be shown.

We also provide the ground truth lidar poses in the KITTI format (one pose is a flattened 3x4 transformation matrix) in the `refsys` folder. These poses can directly be used with [Evo](https://github.com/MichaelGrupp/evo) for fast and easy evaluation of your approaches on our data.

# Data description

## Lidar data
All lidar data can be found inside the `ouster` folder. One point cloud is saved as a single binary file with its file name indicating the Unix time stamp of the start of the revolution. All point clouds contain exactly 128 * 2048 lidar points, with one lidar points having a length of 6 floats, corresponding to its 6 fields [x, y, z, timestamp, reflectivity, near_ir]. To preserve the structure of the lidar scan, we have decided to include missing points through  nan-values in their data fields. The point cloud is sorted by laser-beam ([0,128], from up to down) and time since start of the revolution ([0,1E8], nanoseconds since the start of the lidar scan).

### Lidar data fields
- x: x-coordinate of the point (forward)
- y: y-coordinate of the point (left)
- z: z-coordinate of the point (up)
- t: time in nanoseconds since the start of the revolution [0,1E8]
- reflectivity: reflectivity [0,4096] (TODO: Check these bounds)
- near_ir: near infrared value [0, 255] (TODO: Check these bounds)