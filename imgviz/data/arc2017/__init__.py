# flake8: noqa

import os.path as osp

import numpy as np


here = osp.dirname(osp.abspath(__file__))


def arc2017():
    data_file = osp.join(here, '1532900700692405455.npz')
    data = np.load(data_file)
    data = dict(data)

    # compose masks to class label image
    class_label = np.full(data['rgb'].shape[:2], 0, dtype=np.int32)
    for l, mask in zip(data['labels'], data['masks']):
        class_label[mask == 1] = l
    data['class_label'] = class_label

    names_file = osp.join(here, 'class_names.txt')
    with open(names_file) as f:
        class_names = [name.strip() for name in f]
    data['class_names'] = class_names

    data['res4'] = np.load(osp.join(here, 'res4.npz'))['res4']

    return data
