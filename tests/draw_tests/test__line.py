import numpy as np

import imgviz


def test_line() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.line(img, yx=[(10, 10), (90, 90)], fill=(0, 0, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)
