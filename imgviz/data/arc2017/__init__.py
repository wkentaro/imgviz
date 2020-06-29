# flake8: noqa

import os.path as osp

import numpy as np
import yaml


here = osp.dirname(osp.abspath(__file__))


def arc2017():
    data_file = osp.join(here, "data.npz")
    data = dict(np.load(data_file))

    # compose masks to class label image
    class_label = np.full(data["rgb"].shape[:2], 0, dtype=np.int32)
    for l, mask in zip(data["labels"], data["masks"]):
        class_label[mask == 1] = l
    data["class_label"] = class_label

    names_file = osp.join(here, "class_names.txt")
    with open(names_file) as f:
        class_names = [name.strip() for name in f]
    data["class_names"] = class_names

    data["res4"] = np.load(osp.join(here, "res4.npz"), allow_pickle=True)[
        "res4"
    ]

    with open(osp.join(here, "camera_info.yaml")) as f:
        data["camera_info"] = yaml.safe_load(f)

    return data
