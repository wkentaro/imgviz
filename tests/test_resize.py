import numpy as np

import imgviz


def test_resize():
    img = np.random.uniform(0, 255, size=(15, 25, 3)).round().astype(np.uint8)

    dst = imgviz.resize(img, height=12)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8

    dst = imgviz.resize(img, width=20)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8

    dst = imgviz.resize(img, height=0.8)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8

    dst = imgviz.resize(img, width=0.8)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8
