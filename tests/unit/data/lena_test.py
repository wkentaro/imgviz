import numpy as np

import imgviz


def test_lena() -> None:
    image = imgviz.data.lena()
    assert image.dtype == np.uint8
    assert image.shape == (512, 512, 3)
