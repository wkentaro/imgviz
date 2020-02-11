import os.path as osp

import numpy as np

from ..._io import imread


here = osp.dirname(osp.abspath(__file__))


# Code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy  # NOQA
def read_flow(filename):
    """Read .flo file in Middlebury format"""

    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise IOError("Invalid .flo file: {}".format(filename))

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow_uv = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        flow_uv = np.resize(flow_uv, (h, w, 2))
        return flow_uv


def middlebury():
    # http://vision.middlebury.edu/flow/data/

    rgb_file = osp.join(here, "grove3.png")
    rgb = imread(rgb_file)

    flow_file = osp.join(here, "grove3.flo")
    flow = read_flow(flow_file)

    data = {"rgb": rgb, "flow": flow}
    return data
