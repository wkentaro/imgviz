import numpy as np
import PIL.Image


def imread(filename):
    # type: (str) -> np.ndarray
    return np.asarray(PIL.Image.open(filename))


def imsave(filename, arr):
    # type: (str, np.ndarray) -> None
    return PIL.Image.fromarray(arr).save(filename)
