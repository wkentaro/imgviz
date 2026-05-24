#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

import imgviz


def diff() -> None:
    # Two edits applied to the same image, chosen so each diff mode tells a
    # different story:
    #   * Brighten the left half - a luminance shift that preserves the
    #     local pattern of edges and textures.
    #   * Paste a flat color patch in the lower right - a localized
    #     structural change that wipes out the underlying detail.
    # signed reveals the direction of the change (a brighter or darker
    # than b), abs reveals total magnitude weighting the patch and the
    # brightness shift roughly equally, and ssim downweights the
    # brightness shift relative to the structural patch (the patch
    # destroys local contrast, the shift preserves it).
    a = imgviz.data.arc2017()["rgb"]
    b = a.copy()

    h, w = b.shape[:2]
    half = w // 2
    b[:, :half] = np.clip(b[:, :half].astype(np.int16) + 20, 0, 255).astype(np.uint8)
    py0, px0 = h - 110, w - 110
    b[py0 : py0 + 80, px0 : px0 + 80] = (255, 230, 0)

    signed = imgviz.diff(a, b, mode="signed")
    magnitude = imgviz.diff(a, b, mode="abs")
    structural = imgviz.diff(a, b, mode="ssim")

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(151)
    plt.title("a (original)", fontsize=9)
    plt.imshow(a)
    plt.axis("off")

    plt.subplot(152)
    plt.title("b (edited)", fontsize=9)
    plt.imshow(b)
    plt.axis("off")

    plt.subplot(153)
    plt.title("signed\n(over / under)", fontsize=9)
    plt.imshow(signed)
    plt.axis("off")

    plt.subplot(154)
    plt.title("abs\n(magnitude)", fontsize=9)
    plt.imshow(magnitude)
    plt.axis("off")

    plt.subplot(155)
    plt.title("ssim\n(structure)", fontsize=9)
    plt.imshow(structural)
    plt.axis("off")


if __name__ == "__main__":
    from _base import run_example

    run_example(diff)
