#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def resize():
    data = imgviz.data.arc2017()

    rgb = data["rgb"]

    H, W = rgb.shape[:2]
    rgb_resized = imgviz.resize(rgb, height=0.1)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title(f"rgb:\n{rgb.shape}")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(122)
    plt.title(f"rgb_resized:\n{rgb_resized.shape}")
    plt.imshow(rgb_resized)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(resize)
