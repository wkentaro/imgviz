#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def rounded_rectangle() -> None:
    img = imgviz.data.lena()
    height, width = img.shape[:2]
    colors = imgviz.label_colormap()[1:]

    radii = [0, 40, 100]
    box_height = height * 0.5
    box_width = width * 0.22

    viz = img
    for i, radius in enumerate(radii):
        cy = height / 2
        cx = width * (i + 1) / (len(radii) + 1)
        color = colors[i]
        y1, x1 = cy - box_height / 2, cx - box_width / 2
        y2, x2 = cy + box_height / 2, cx + box_width / 2
        viz = imgviz.draw.rounded_rectangle(
            viz,
            yx1=(y1, x1),
            yx2=(y2, x2),
            radius=radius,
            outline=(int(color[0]), int(color[1]), int(color[2])),
            width=5,
        )
        viz = imgviz.draw.text_in_rectangle(
            viz,
            loc="lt+",
            text=f"radius={radius}",
            size=18,
            background=(int(color[0]), int(color[1]), int(color[2])),
            yx1=(y1, x1),
            yx2=(y2, x2),
        )

    plt.figure(dpi=200)
    plt.imshow(viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(rounded_rectangle)
