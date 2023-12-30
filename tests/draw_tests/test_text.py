import numpy as np

import imgviz


def test_text():
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.text(img, yx=(0, 0), text="TEST", color=(0, 0, 0), size=30)
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.allclose(img, res)
