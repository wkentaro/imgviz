#!/usr/bin/env python

import typing
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import imgviz


def label2rgb() -> None:
    data = imgviz.data.voc()

    rgb: NDArray[np.uint8] = data["rgb"]
    label: NDArray[np.int32] = data["class_label"]
    label_names: list[str] = [f"{i}:{n}" for i, n in enumerate(data["class_names"])]

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("+img")
    plt.imshow(imgviz.label2rgb(label=label, image=imgviz.rgb2gray(rgb)))
    plt.axis("off")

    kwargs: dict = dict(label_names=label_names, font_size=25)

    for i, loc in enumerate(["centroid", "rb"]):
        loc = typing.cast(Literal["centroid", "rb"], loc)
        plt.subplot(132 + i)
        plt.title(f"loc={loc}")
        plt.imshow(imgviz.label2rgb(label, loc=loc, **kwargs))
        plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(label2rgb)
