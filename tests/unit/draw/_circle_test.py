import numpy as np
import pytest

import imgviz


def test_circle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.circle(img, center=(50, 50), diameter=30, fill=(0, 0, 255))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)


def test_circle_rejects_missing_fill_and_outline() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.circle(img, center=(50, 50), diameter=30)
