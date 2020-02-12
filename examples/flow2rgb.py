#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def flow2rgb():
    data = imgviz.data.middlebury()

    rgb = data["rgb"]
    flowviz = imgviz.flow2rgb(data["flow"])

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title("image")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(122)
    plt.title("flow")
    plt.imshow(flowviz)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(flow2rgb)
