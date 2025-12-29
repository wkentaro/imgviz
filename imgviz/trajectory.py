from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ._io import pyplot_to_numpy
from .external import transformations as tf


def plot_trajectory(
    transforms: NDArray,
    is_relative: bool = False,
    mode: Literal["xy", "xz", "yx", "yz", "zx", "zy"] = "xz",
    style: str = "b.",
    axis: bool = True,
) -> NDArray[np.uint8]:
    """Plot the trajectory using transform matrices.

    Parameters
    ----------
    transforms
        Transform matrices with shape (N, 4, 4) where N is the number of poses.
    is_relative
        True for relative poses.
    mode
        X and Y axis of trajectory (e.g., 'xz' following KITTI format).
    style
        Style of plotting.
    axis
        False to disable axis.

    Returns
    -------
    dst
        Trajectory image.

    """
    import matplotlib.pyplot as plt  # slow

    if is_relative:
        for i in range(1, len(transforms)):
            transforms[i] = transforms[i - 1].dot(transforms[i])

    if len(mode) != 2 and all(x in "xyz" for x in mode):
        raise ValueError(f"Unsupported mode: {mode}")

    x = []
    y = []
    index_x = "xyz".index(mode[0])
    index_y = "xyz".index(mode[1])
    for T in transforms:
        translate = tf.translation_from_matrix(T)
        x.append(translate[index_x])
        y.append(translate[index_y])

    # swith backend to agg for supporting no display mode
    backend = plt.get_backend()
    plt.switch_backend("agg")

    plt.plot(x, y, style)

    if not axis:
        plt.axis("off")

    dst = pyplot_to_numpy()
    plt.close()

    # switch back backend
    plt.switch_backend(backend)

    return dst
