import numpy as np
import pytest

import imgviz


def test_circle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.circle(img, center=(50, 50), diameter=30, fill=(0, 0, 255))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)


def test_circle_center_is_yx_not_xy() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    center = (30, 70)
    res = imgviz.draw.circle(img, center=center, diameter=20, fill=(0, 0, 255))
    cy, cx = center
    assert (res[cy, cx] == (0, 0, 255)).all()
    assert (res[cx, cy] == (255, 255, 255)).all()


def test_circle_fills_disc_of_diameter() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.circle(img, center=(30, 70), diameter=20, fill=(0, 0, 255))
    assert (res[30, 78] == (0, 0, 255)).all()
    assert (res[30, 82] == (255, 255, 255)).all()
    assert (res[38, 70] == (0, 0, 255)).all()
    assert (res[42, 70] == (255, 255, 255)).all()
    assert (res[22, 62] == (255, 255, 255)).all()


def test_circle_rejects_missing_fill_and_outline() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.circle(img, center=(50, 50), diameter=30)
