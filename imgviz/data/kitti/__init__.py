import os.path as osp

import numpy as np

here = osp.dirname(osp.abspath(__file__))


def read_pose_file(filename):
    with open(filename, "r") as f:
        transforms = []
        for one_line in f:
            one_line = one_line.split(" ")
            Rt = [float(pose) for pose in one_line] + [0, 0, 0, 1]

            Rt = np.reshape(np.array(Rt), (4, 4))
            assert abs(Rt[3].sum() - 1) < 1e-5
            transforms.append(Rt)
    return transforms


def kitti_odometry():
    # http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    pose_file = osp.join(here, "odometry/00.txt")
    transforms = read_pose_file(pose_file)
    data = {"transforms": transforms}
    return data
