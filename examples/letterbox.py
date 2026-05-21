#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def letterbox() -> None:
    data = imgviz.data.arc2017()

    rgb = data["rgb"]

    H, W = rgb.shape[:2]
    letterboxed1 = imgviz.letterbox(rgb, height=H, width=H)

    rgb_T = rgb.transpose(1, 0, 2)
    letterboxed2 = imgviz.letterbox(rgb_T, height=H, width=H)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("original")
    plt.axis("off")
    plt.imshow(rgb)

    plt.subplot(132)
    plt.title("letterboxed1")
    plt.imshow(letterboxed1)
    plt.axis("off")

    plt.subplot(133)
    plt.title("letterboxed2")
    plt.imshow(letterboxed2)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(letterbox)
