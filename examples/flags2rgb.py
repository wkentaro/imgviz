#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import imgviz


def flags2rgb() -> None:
    data = imgviz.data.arc2017()

    masks = data["masks"] == 1
    bboxes = data["bboxes"]
    centers = np.array([np.argwhere(mask).mean(axis=0) for mask in masks])
    flags = np.column_stack(
        (
            masks.sum(axis=(1, 2)) < 7000,
            (bboxes[:, 2] - bboxes[:, 0]) > (bboxes[:, 3] - bboxes[:, 1]),
            centers[:, 1] < data["rgb"].shape[1] / 2,
        )
    )
    flag_names = ["small", "tall", "left"]

    flagviz1 = imgviz.flags2rgb(
        data["rgb"], flags=flags, centers=centers, flag_names=flag_names
    )
    flagviz2 = imgviz.flags2rgb(
        data["rgb"], flags=flags, centers=centers, flag_names=flag_names, wedges="all"
    )

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("rgb")
    plt.imshow(data["rgb"])
    plt.axis("off")

    plt.subplot(132)
    plt.title('flags (wedges="on")')
    plt.imshow(flagviz1)
    plt.axis("off")

    plt.subplot(133)
    plt.title('flags (wedges="all")')
    plt.imshow(flagviz2)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(flags2rgb)
