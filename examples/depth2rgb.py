#!/usr/bin/env python

import cmap
import matplotlib.pyplot as plt

import imgviz


def depth2rgb() -> None:
    data = imgviz.data.arc2017()

    depthviz_jet = imgviz.depth2rgb(
        data["depth"], min_value=0.3, max_value=1, colormap="jet"
    )

    colormap = cmap.Colormap([(0, 0, 0), (0, 255, 0)])
    depthviz_custom = imgviz.depth2rgb(
        data["depth"], min_value=0.3, max_value=1, colormap=colormap
    )

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("rgb")
    plt.imshow(data["rgb"])
    plt.axis("off")

    plt.subplot(132)
    plt.title("depth (jet color)")
    plt.imshow(depthviz_jet)
    plt.axis("off")

    plt.subplot(133)
    plt.title("depth (custom color)")
    plt.imshow(depthviz_custom)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(depth2rgb)
