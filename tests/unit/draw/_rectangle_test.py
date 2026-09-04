import numpy as np
import pytest

import imgviz


def test_rectangle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.rectangle(img, (0, 0), (50, 50), outline=(0, 0, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype


def test_rectangle_corners_are_yx_not_xy() -> None:
    img = np.full((80, 80, 3), 255, dtype=np.uint8)
    res = imgviz.draw.rectangle(img, yx1=(10, 20), yx2=(30, 60), fill=(0, 0, 0))
    assert tuple(res[20, 40]) == (0, 0, 0)
    assert tuple(res[40, 25]) == (255, 255, 255)


def test_rectangle_rejects_missing_fill_and_outline() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.rectangle(img, (0, 0), (50, 50))
