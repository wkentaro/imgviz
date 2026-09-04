import numpy as np
import pytest

import imgviz


def test_star() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.star(img, center=(50, 50), size=20, fill=(255, 0, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)


def test_star_spikes_are_solid_with_empty_notches() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.star(img, center=(50, 50), size=60, fill=(255, 0, 0))
    filled = np.all(res == (255, 0, 0), axis=2)
    # the top spike is filled solid, from its tip straight down through the center
    assert filled[22:51, 50].all()
    # a notch-free blob (e.g. a decagon) would fill these flanking points;
    # the star's concave notches leave them empty
    assert not filled[32, 37]
    assert not filled[32, 63]


def test_star_rejects_missing_fill_and_outline() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.star(img, center=(50, 50), size=20)
