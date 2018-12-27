# flake8: noqa

import os.path as osp

import numpy as np


here = osp.dirname(osp.abspath(__file__))


def arc2017():
    data_file = osp.join(here, '1532900700692405455.npz')
    data = np.load(data_file)
    return dict(data)
