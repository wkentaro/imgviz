#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def box_corners() -> None:
    img = imgviz.data.lena()
    height, width = img.shape[:2]
    colors = imgviz.label_colormap()[1:]

    boxes = [
        ((height * 0.12, width * 0.12), (height * 0.52, width * 0.52)),
        ((height * 0.45, width * 0.45), (height * 0.88, width * 0.88)),
    ]

    viz = img
    for (yx1, yx2), color in zip(boxes, colors):
        viz = imgviz.draw.box_corners(
            viz,
            yx1=yx1,
            yx2=yx2,
            fill=color,
            length=30,
            width=4,
        )

    plt.figure(dpi=200)
    plt.imshow(viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(box_corners)
