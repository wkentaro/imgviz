#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def centerize():
    data = imgviz.data.arc2017()

    rgb = data["rgb"]

    H, W = rgb.shape[:2]
    centerized1 = imgviz.centerize(rgb, shape=(H, H))

    rgb_T = rgb.transpose(1, 0, 2)
    centerized2 = imgviz.centerize(rgb_T, shape=(H, H))

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("original")
    plt.axis("off")
    plt.imshow(rgb)

    plt.subplot(132)
    plt.title(f"centerized1:\n{centerized1.shape}")
    plt.imshow(centerized1)
    plt.axis("off")

    plt.subplot(133)
    plt.title(f"centerized2:\n{centerized2.shape}")
    plt.imshow(centerized2)
    plt.axis("off")

    return imgviz.io.pyplot_to_numpy()


if __name__ == "__main__":
    from base import run_example

    run_example(centerize)
