import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import imgviz


def test_mask2rgb_mask_only(show: bool) -> None:
    COLOR_RED: tuple[int, int, int] = (255, 0, 0)

    mask: NDArray[np.bool_] = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True

    result = imgviz.mask2rgb(mask, color=COLOR_RED)
    if show:
        plt.imshow(result)
        plt.show()

    assert result.dtype == np.uint8
    assert result.shape == (10, 10, 3)
    assert (result[mask] == COLOR_RED).all()
    assert (result[~mask] == (0, 0, 0)).all()


def test_mask2rgb_with_image(show: bool) -> None:
    COLOR_WHITE: tuple[int, int, int] = (255, 255, 255)

    mask: NDArray[np.bool_] = np.zeros((10, 10), dtype=bool)
    mask[2:8, 2:8] = True
    image = np.full((10, 10, 3), COLOR_WHITE, dtype=np.uint8)

    result = imgviz.mask2rgb(mask, image=image, alpha=0.5)
    if show:
        plt.imshow(result)
        plt.show()

    assert (result[mask] == (128, 255, 128)).all()
    assert (result[~mask] == COLOR_WHITE).all()
