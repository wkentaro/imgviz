#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import imgviz


def nchannel2rgb():
    data = imgviz.data.arc2017()

    nchannel_viz = imgviz.nchannel2rgb(data["res4"], dtype=np.float32)

    H, W = data["rgb"].shape[:2]
    nchannel_viz = imgviz.resize(nchannel_viz, height=H, width=W)
    nchannel_viz = (nchannel_viz * 255).round().astype(np.uint8)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title("rgb")
    plt.imshow(data["rgb"])
    plt.axis("off")

    plt.subplot(122)
    plt.title("res4 (colorized)")
    plt.imshow(nchannel_viz)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(nchannel2rgb)
