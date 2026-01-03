from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray

from ...io import imread

_here: pathlib.Path = pathlib.Path(__file__).parent


def lena() -> NDArray[np.uint8]:
    image_file = _here / "lena.png"
    image = imread(image_file)
    return image
