import numpy as np

import imgviz


def test_tile():
    img1 = np.random.uniform(0, 255, (15, 25, 3)).round().astype(np.uint8)
    img2 = np.random.uniform(0, 255, (25, 25, 3)).round().astype(np.uint8)
    img3 = np.random.uniform(0, 255, (25, 10, 3)).round().astype(np.uint8)
    tiled = imgviz.tile([img1, img2, img3], (1, 3))

    assert tiled.shape == (25, 75, 3)
    assert tiled.dtype == np.uint8
