#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def instances2rgb() -> None:
    data = imgviz.data.voc()

    captions = [data["class_names"][label_id] for label_id in data["labels"]]
    insviz1 = imgviz.instances2rgb(
        image=data["rgb"],
        bboxes=data["bboxes"],
        labels=data["labels"],
        captions=captions,
    )
    insviz2 = imgviz.instances2rgb(
        image=data["rgb"],
        masks=data["masks"] == 1,
        labels=data["labels"],
        captions=captions,
    )

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(131)
    plt.title("rgb")
    plt.imshow(data["rgb"])
    plt.axis("off")

    plt.subplot(132)
    plt.title("instances (bboxes)")
    plt.imshow(insviz1)
    plt.axis("off")

    plt.subplot(133)
    plt.title("instances (masks)")
    plt.imshow(insviz2)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(instances2rgb)
