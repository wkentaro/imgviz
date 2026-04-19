#!/usr/bin/env python

import cmap
import matplotlib.pyplot as plt

import imgviz


def colorize() -> None:
    data = imgviz.data.arc2017()

    depthviz_viridis = imgviz.colorize(data["depth"], vmin=0.3, vmax=1, cmap="viridis")

    custom_cmap = cmap.Colormap([(0, 0, 0), (0, 255, 0)])
    depthviz_custom = imgviz.colorize(data["depth"], vmin=0.3, vmax=1, cmap=custom_cmap)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("rgb")
    plt.imshow(data["rgb"])
    plt.axis("off")

    plt.subplot(132)
    plt.title("depth (viridis)")
    plt.imshow(depthviz_viridis)
    plt.axis("off")

    plt.subplot(133)
    plt.title("depth (custom)")
    plt.imshow(depthviz_custom)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(colorize)
