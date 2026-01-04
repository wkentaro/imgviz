#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import imgviz


def nchannel2rgb() -> None:
    data = imgviz.data.arc2017()

    nchannel_viz: NDArray[np.uint8] = imgviz.nchannel2rgb(data["res4"])

    height, width = data["rgb"].shape[:2]
    nchannel_viz = imgviz.resize(nchannel_viz, height=height, width=width)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title("rgb")
    plt.imshow(data["rgb"])
    plt.axis("off")

    plt.subplot(122)
    plt.title("res4 (colorized)")
    plt.imshow(nchannel_viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(nchannel2rgb)
