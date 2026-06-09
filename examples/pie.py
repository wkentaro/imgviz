#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def pie() -> None:
    img = imgviz.data.lena()
    height, width = img.shape[:2]
    colors = imgviz.label_colormap()[1:]

    viz = img
    for n_wedges in range(1, 6):
        viz = imgviz.draw.pie(
            viz,
            center=(height / 2, width * n_wedges / 6),
            diameter=70,
            fills=[
                (int(color[0]), int(color[1]), int(color[2]))
                for color in colors[:n_wedges]
            ],
            outline=(255, 255, 255),
            width=2,
        )

    plt.figure(dpi=200)
    plt.imshow(viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(pie)
