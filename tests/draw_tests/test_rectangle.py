import numpy as np

import imgviz


def test_rectangle():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.rectangle(img, (0, 0), (50, 50), outline=(0, 0, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
