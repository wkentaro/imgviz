import numpy as np

import imgviz


def test_ellipse() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.ellipse(img, aabb1=(20, 20), aabb2=(80, 60), fill=(0, 255, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)
