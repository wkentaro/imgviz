#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def tint() -> None:
    img = imgviz.data.arc2017()["rgb"]
    panels = [
        ("original", img),
        ("red", imgviz.tint(img, "red")),
        ("green", imgviz.tint(img, "green")),
        ("blue", imgviz.tint(img, "blue")),
    ]

    plt.figure(dpi=200)
    for i, (title, panel) in enumerate(panels):
        plt.subplot(1, 4, i + 1)
        plt.title(title)
        plt.imshow(panel)
        plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(tint)
