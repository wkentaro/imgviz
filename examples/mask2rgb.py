#!/usr/bin/env python

import functools
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import imgviz


def mask2rgb():
    data: dict[str, Any] = imgviz.data.voc()

    rgb: NDArray[np.uint8] = data["rgb"]
    masks: NDArray[np.bool_] = data["masks"]
    labels: NDArray[np.int32] = data["labels"]
    class_names: list[str] = data["class_names"]

    keep: NDArray[np.bool_] = labels == class_names.index("diningtable")
    mask: NDArray[np.bool_] = functools.reduce(np.logical_or, masks[keep])

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("mask only")
    plt.imshow(imgviz.mask2rgb(mask))
    plt.axis("off")

    plt.subplot(132)
    plt.title("mask + image")
    plt.imshow(imgviz.mask2rgb(mask, image=rgb, alpha=0.5))
    plt.axis("off")

    plt.subplot(133)
    plt.title("mask + image\n(color=red)")
    plt.imshow(imgviz.mask2rgb(mask, image=rgb, alpha=0.5, color=(255, 0, 0)))
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(mask2rgb)
