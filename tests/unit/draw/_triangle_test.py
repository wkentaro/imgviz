import numpy as np
import pytest

import imgviz


def test_triangle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.triangle(img, center=(50, 50), size=20, fill=(0, 0, 255))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)


def test_triangle_apex_points_up() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.triangle(img, center=(50, 50), size=60, fill=(255, 0, 0))
    filled = np.all(res == (255, 0, 0), axis=2)
    # a single apex at the top center, widening into a flat base at the bottom
    assert filled[22, 45:56].any()
    assert filled[25].sum() < filled[60].sum()
    # the base is the bottom edge; nothing extends below it
    assert not filled[67, 45:56].any()


def test_triangle_rejects_missing_fill_and_outline() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.triangle(img, center=(50, 50), size=20)
