#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def scalebar() -> None:
    img = imgviz.data.arc2017()["rgb"]
    viz = imgviz.scalebar(img, pixels_per_unit=12.8, unit="cm", loc="rb")

    plt.figure(dpi=200)
    plt.imshow(viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(scalebar)
