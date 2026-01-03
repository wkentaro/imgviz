#!/usr/bin/env python

import typing

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import imgviz


def flow2rgb() -> None:
    data = imgviz.data.middlebury()

    assert isinstance(data["rgb"], np.ndarray)
    assert data["rgb"].dtype == np.uint8
    rgb: NDArray[np.uint8] = typing.cast(NDArray[np.uint8], data["rgb"])

    assert isinstance(data["flow"], np.ndarray)
    assert data["flow"].dtype == np.float32
    flow: NDArray[np.float32] = typing.cast(NDArray[np.float32], data["flow"])
    flowviz = imgviz.flow2rgb(flow)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title("image")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(122)
    plt.title("flow")
    plt.imshow(flowviz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(flow2rgb)
