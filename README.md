# Odyssey Devkit

Odyssey is a dataset taylored towards lidar-inertial-odometry (LIO). This repository contains all accompanying code, including the Python dataloader as well as examples for its usage.

Download the dataset from our homepage [https://odyssey.uni-goettingen.de/](https://odyssey.uni-goettingen.de/)

Read our paper preprint on [arXiv](TODO)

# Python Dependencies
- Numpy
- Scipy
- Matplotlib

# Quickstart
Clone this repo with
```bash
git clone git@github.com:Fusion-Goettingen/odyssey-devkit.git
cd odyssey-devkit
```
modify the `base_dir` and `seq` in `example.py` to point to the directory of the Odyssey dataset and execute with
```bash
python3 example.py
```
to see our dataloader in action.