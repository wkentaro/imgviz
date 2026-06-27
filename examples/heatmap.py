#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import imgviz


def heatmap() -> None:
    img = imgviz.data.lena()
    height, width = img.shape[:2]

    rng = np.random.default_rng(0)
    centers = [(height * 0.35, width * 0.4), (height * 0.6, width * 0.65)]
    points = np.concatenate(
        [rng.normal(loc=c, scale=40, size=(150, 2)) for c in centers]
    )

    density = imgviz.heatmap(points, shape=(height, width), sigma=15)
    colored = imgviz.colorize(density, cmap="turbo")
    overlay = (0.5 * img + 0.5 * colored).astype(np.uint8)

    plt.figure(dpi=200)

    plt.subplot(1, 2, 1)
    plt.title("points")
    plt.imshow(img)
    plt.scatter(points[:, 1], points[:, 0], s=2, c="white")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("heatmap")
    plt.imshow(overlay)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(heatmap)
