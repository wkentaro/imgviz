import numpy as np
import pytest

import imgviz


def test_rectangle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.rectangle(img, (0, 0), (50, 50), outline=(0, 0, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype


def test_rectangle_rejects_missing_fill_and_outline() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.rectangle(img, (0, 0), (50, 50))
