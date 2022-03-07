import os
import os.path as osp

import numpy as np
import PIL.Image

from .. import utils


def imread(filename):
    # type: (str) -> np.ndarray
    """Read image from file.

    Parameters
    ----------
    filename: str
        Filename.

    Returns
    -------
    img: numpy.ndarray, (H, W) or (H, W, 3) or (H, W, 4)
        Image read.
    """
    return utils.pillow_to_numpy(PIL.Image.open(filename))


def imsave(filename, arr):
    # type: (str, np.ndarray) -> None
    """Save image to file.

    Parameters
    ----------
    filename: str
        Filename.
    arr: numpy.ndarray, (H, W) or (H, W, 3) or (H, W, 4)
        Image to save.

    Returns
    -------
    None

    """
    try:
        os.makedirs(osp.dirname(filename))
    except OSError:
        pass
    return utils.numpy_to_pillow(arr).save(filename)
