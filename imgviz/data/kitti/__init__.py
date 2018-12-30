import os.path as osp

import numpy as np


here = osp.dirname(osp.abspath(__file__))


def read_pose_file(filename):
    with open(filename, 'r') as f:
        transforms = []
        for one_line in f:
            one_line = one_line.split(' ')
            Rt = [float(pose) for pose in one_line] + [0, 0, 0, 1]

            Rt = np.reshape(np.array(Rt), (4, 4))
            assert abs(Rt[3].sum() - 1) < 1e-5
            transforms.append(Rt)
    return transforms


def read_flow(filename):
    """Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-
    # flow-files-with-python-bytes-array-numpy

    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            print('Reading {0} x {1} flo file'.format(w, h))
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data = np.resize(data, (h, w, 2))

            return data


def kitti():
    pose_file = osp.join(here, 'v0/00.txt')
    transforms = read_pose_file(pose_file)

    flow_file = osp.join(here, 'flow/flow.flo')
    flow = read_flow(flow_file)

    origin_file = osp.join(here, 'flow/frame.npy')
    origin = np.load(origin_file)

    data = {'transforms': transforms, 'flow': (origin, flow)}
    return data
