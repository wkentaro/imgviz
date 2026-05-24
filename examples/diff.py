#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import skimage.filters

import imgviz


def diff() -> None:
    # Evaluating a distance-transform regressor (the target of watershed-style
    # instance segmentation): the model predicts, per pixel, the distance to
    # the object boundary. Here the ground truth is the true distance field and
    # the prediction stands in for a model that under-predicts large distances
    # (object interiors are hard) on top of some low-frequency noise.
    masks = imgviz.data.arc2017()["masks"]
    foreground = (masks > 0).any(axis=0)
    ground_truth = scipy.ndimage.distance_transform_edt(foreground).astype(np.float32)

    low_frequency_noise = skimage.filters.gaussian(
        np.random.RandomState(0).randn(*ground_truth.shape).astype(np.float32),
        sigma=12,
        preserve_range=True,
    )
    low_frequency_noise *= 6.0 / (np.abs(low_frequency_noise).max() + 1e-8)
    # Concentrate the error on the objects (the easy background stays exact),
    # feathering it to zero across the boundary so there is no hard ring.
    inside = skimage.filters.gaussian(foreground.astype(np.float32), sigma=4)
    error = (-0.25 * ground_truth + low_frequency_noise) * inside
    prediction = np.clip(ground_truth + error, 0, None)

    signed = imgviz.diff(ground_truth, prediction, mode="signed")
    magnitude = imgviz.diff(ground_truth, prediction, mode="abs")
    structural = imgviz.diff(ground_truth, prediction, mode="ssim")

    vmax = float(ground_truth.max())

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(151)
    plt.title("ground truth\n(target)", fontsize=9)
    plt.imshow(imgviz.colorize(ground_truth, vmin=0, vmax=vmax))
    plt.axis("off")

    plt.subplot(152)
    plt.title("prediction\n(model)", fontsize=9)
    plt.imshow(imgviz.colorize(prediction, vmin=0, vmax=vmax))
    plt.axis("off")

    plt.subplot(153)
    plt.title("signed\n(over / under)", fontsize=9)
    plt.imshow(signed)
    plt.axis("off")

    plt.subplot(154)
    plt.title("abs\n(error size)", fontsize=9)
    plt.imshow(magnitude)
    plt.axis("off")

    plt.subplot(155)
    plt.title("ssim\n(structure)", fontsize=9)
    plt.imshow(structural)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(diff)
