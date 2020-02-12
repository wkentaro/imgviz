#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def tile():
    data = imgviz.data.arc2017()

    rgb = data["rgb"]
    bboxes = data["bboxes"].astype(int)
    masks = data["masks"] == 1
    crops = []
    for bbox, mask in zip(bboxes, masks):
        slice_ = slice(bbox[0], bbox[2]), slice(bbox[1], bbox[3])
        rgb_crop = rgb[slice_]
        mask_crop = mask[slice_]
        crops.append(rgb_crop * mask_crop[:, :, None])
    tiled = imgviz.tile(imgs=crops, border=(255, 255, 255))

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title("original")
    plt.imshow(rgb)
    plt.axis("off")

    plt.subplot(122)
    plt.title("instances")
    plt.imshow(tiled)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(tile)
