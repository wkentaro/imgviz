from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray

_here: pathlib.Path = pathlib.Path(__file__).parent


def read_pose_file(filename: str | pathlib.Path) -> list[NDArray[np.float64]]:
    with open(filename) as f:
        transforms = []
        for one_line in f:
            one_line = one_line.split(" ")
            Rt = [float(pose) for pose in one_line] + [0, 0, 0, 1]

            Rt = np.reshape(np.array(Rt), (4, 4))
            if abs(Rt[3].sum() - 1) >= 1e-5:
                raise ValueError(
                    f"invalid pose: last row should be [0, 0, 0, 1], got {Rt[3]}"
                )
            transforms.append(Rt)
    return transforms


def kitti_odometry() -> dict[str, list[NDArray[np.float64]]]:
    # http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    pose_file = _here / "odometry/00.txt"
    transforms = read_pose_file(pose_file)
    data = {"transforms": transforms}
    return data
