#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def label2rgb():
    data = imgviz.data.voc()

    rgb = data["rgb"]
    label = data["class_label"]

    label_names = [f"{i}:{n}" for i, n in enumerate(data["class_names"])]
    labelviz_withname1 = imgviz.label2rgb(
        label, label_names=label_names, font_size=25, loc="centroid"
    )
    labelviz_withname2 = imgviz.label2rgb(
        label, label_names=label_names, font_size=25, loc="rb"
    )
    img = imgviz.color.rgb2gray(rgb)
    labelviz_withimg = imgviz.label2rgb(label=label, image=img)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("+img")
    plt.imshow(labelviz_withimg)
    plt.axis("off")

    plt.subplot(132)
    plt.title("loc=centroid")
    plt.imshow(labelviz_withname1)
    plt.axis("off")

    plt.subplot(133)
    plt.title("loc=rb")
    plt.imshow(labelviz_withname2)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(label2rgb)
