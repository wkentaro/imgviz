#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def resize() -> None:
    data = imgviz.data.arc2017()

    rgb = data["rgb"]

    H, W = rgb.shape[:2]
    rgb_resized = imgviz.resize(rgb, height=0.1)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title(f"original: {rgb.shape[0]}x{rgb.shape[1]}")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(122)
    plt.title(f"resized: {rgb_resized.shape[0]}x{rgb_resized.shape[1]}")
    plt.imshow(rgb_resized)
    plt.axis("off")


if __name__ == "__main__":
    from base import run_example

    run_example(resize)
