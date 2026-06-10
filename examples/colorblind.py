#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def colorblind() -> None:
    img = imgviz.data.arc2017()["rgb"]
    panels = [
        ("original", img),
        ("protanopia", imgviz.colorblind(img, kind="protanopia")),
        ("deuteranopia", imgviz.colorblind(img, kind="deuteranopia")),
        ("tritanopia", imgviz.colorblind(img, kind="tritanopia")),
    ]

    plt.figure(dpi=200)
    for i, (title, panel) in enumerate(panels):
        plt.subplot(2, 2, i + 1)
        plt.title(title)
        plt.imshow(panel)
        plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(colorblind)
