import pathlib

import numpy as np
import PIL.Image
from numpy.typing import NDArray

from . import _utils
from ._label import label_colormap


def imread(filename: str | pathlib.Path) -> np.ndarray:
    """Read image from file.

    Parameters
    ----------
    filename: str | pathlib.Path
        Filename.

    Returns
    -------
    img: numpy.ndarray, (H, W) or (H, W, 3) or (H, W, 4)
        Image read.
    """
    return _utils.pillow_to_numpy(PIL.Image.open(filename))


def imsave(filename: str | pathlib.Path, arr: np.ndarray) -> None:
    """Save image to file.

    Parameters
    ----------
    filename: str | pathlib.Path
        Filename.
    arr: numpy.ndarray, (H, W) or (H, W, 3) or (H, W, 4)
        Image to save.

    Returns
    -------
    None

    """
    path = pathlib.Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    _utils.numpy_to_pillow(arr).save(path)


def lblsave(filename: str | pathlib.Path, lbl: np.ndarray) -> None:
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
    colormap = label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(filename)


def imshow(image: NDArray[np.uint8]) -> None:
    _utils.numpy_to_pillow(image).show()
