#!/usr/bin/env python

import imgviz


def plot_trajectory():
    data = imgviz.data.kitti_odometry()

    img = imgviz.plot_trajectory(data["transforms"])

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(plot_trajectory)
