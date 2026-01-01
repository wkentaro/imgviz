#!/usr/bin/env python

from typing import Literal
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import imgviz


def label2rgb() -> NDArray[np.uint8]:
    data: dict[str, NDArray] = imgviz.data.voc()

    rgb: NDArray[np.uint8] = data["rgb"]
    label: NDArray[np.int32] = data["class_label"]
    label_names: list[str] = [f"{i}:{n}" for i, n in enumerate(data["class_names"])]

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(231)
    plt.title("+img")
    plt.imshow(imgviz.label2rgb(label=label, image=imgviz.rgb2gray(rgb)))
    plt.axis("off")

    kwargs: dict = dict(label_names=label_names, font_size=25)

    Loc: TypeAlias = Literal["centroid", "lt", "rt", "lb", "rb"]
    for i, loc in enumerate(Loc.__args__):
        plt.subplot(232 + i)
        plt.title(f"loc={loc}")
        plt.imshow(imgviz.label2rgb(label, loc=loc, **kwargs))
        plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(label2rgb)
