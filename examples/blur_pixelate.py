#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import imgviz


def blur_pixelate() -> None:
    rgb = imgviz.data.arc2017()["rgb"]

    H, W = rgb.shape[:2]
    mask = np.zeros((H, W), dtype=bool)
    mask[int(H * 0.30) : int(H * 0.70), int(W * 0.30) : int(W * 0.70)] = True

    blurred = imgviz.blur(rgb, sigma=16.0, mask=mask)
    pixelated = imgviz.pixelate(rgb, block=20, mask=mask)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("original")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(132)
    plt.title("blur (mask)")
    plt.imshow(blurred)
    plt.axis("off")

    plt.subplot(133)
    plt.title("pixelate (mask)")
    plt.imshow(pixelated)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(blur_pixelate)
