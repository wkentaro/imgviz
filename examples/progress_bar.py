#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def progress_bar() -> None:
    img = imgviz.data.lena()
    _, width = img.shape[:2]
    values = [0.15, 0.4, 0.65, 0.9]

    viz = img
    for i, value in enumerate(values):
        y1 = 40 + i * 60
        viz = imgviz.draw.progress_bar(
            viz,
            yx1=(y1, 40),
            yx2=(y1 + 36, width - 40),
            value=value,
            fill=(0, 200, 0),
            background=(0, 0, 0),
            outline=(255, 255, 255),
        )
        viz = imgviz.draw.text(
            viz,
            yx=(y1 + 8, 52),
            text=f"{round(value * 100)}%",
            size=22,
            color=(255, 255, 255),
        )

    plt.figure(dpi=200)
    plt.imshow(viz)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(progress_bar)
