import os
import os.path as osp
import pathlib
from typing import Union

import numpy as np  # NOQA
import PIL.Image

from .. import utils
from ..label import label_colormap


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


def lblsave(filename: Union[str, pathlib.Path], lbl: np.ndarray) -> None:
    """Save label image to PNG file with a colormap.

    Parameters
    ----------
    filename: str | pathlib.Path
        Filename. Must end with '.png'.
    lbl: numpy.ndarray, (H, W), np.uint8
        Label image to save.

    Returns
    -------
    None

    """
    if not str(filename).lower().endswith(".png"):
        raise ValueError(f"filename must end with '.png': {filename}")
    if lbl.dtype != np.uint8:
        raise ValueError(f"lbl.dtype must be np.uint8, but got {lbl.dtype}")

    lbl_pil = PIL.Image.fromarray(lbl, mode="P")
    colormap = label_colormap(n_label=256)
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(filename)
