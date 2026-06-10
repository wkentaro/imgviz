#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def rotated_rectangle() -> None:
    img = imgviz.data.lena()
    height, width = img.shape[:2]
    colors = imgviz.label_colormap()[1:]

    angles = [0, 30, 60]
    box_height = height * 0.5
    box_width = width * 0.22

    viz = img
    for i, angle in enumerate(angles):
        cy = height / 2
        cx = width * (i + 1) / (len(angles) + 1)
        color = colors[i]
        viz = imgviz.draw.rotated_rectangle(
            viz,
            center=(cy, cx),
            size=(box_height, box_width),
            angle=angle,
            outline=(int(color[0]), int(color[1]), int(color[2])),
            width=5,
        )
        viz = imgviz.draw.text_in_rectangle(
            viz,
            loc="lt+",
            text=f"angle={angle}",
            size=18,
            background=(int(color[0]), int(color[1]), int(color[2])),
            yx1=(cy - box_height / 2, cx - box_width / 2),
            yx2=(cy + box_height / 2, cx + box_width / 2),
        )

    plt.figure(dpi=200)
    plt.imshow(viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(rotated_rectangle)
