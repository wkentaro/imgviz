import numpy as np
import PIL.Image
import pytest

import imgviz


def test_arrow() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.arrow(img, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)


def test_arrow_adds_head_over_plain_line() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    line = imgviz.draw.line(img, yx=[(20, 20), (20, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(img, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    changed_by_head = np.any(line != arrow, axis=2)
    assert changed_by_head.sum() > 0


def test_arrow_head_is_at_the_tip() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    line = imgviz.draw.line(img, yx=[(20, 20), (20, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(img, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    ys, xs = np.where(np.any(line != arrow, axis=2))
    assert xs.mean() > 50  # closer to the tip (x=80) than the tail (x=20)


def test_arrow_head_length_ratio_scales_head() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    small = imgviz.draw.arrow(
        img, yx1=(50, 10), yx2=(50, 90), fill=(255, 0, 0), head_length_ratio=0.1
    )
    large = imgviz.draw.arrow(
        img, yx1=(50, 10), yx2=(50, 90), fill=(255, 0, 0), head_length_ratio=0.3
    )
    assert np.any(small != img, axis=2).sum() < np.any(large != img, axis=2).sum()


def test_arrow_head_is_at_the_tip_diagonal() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    line = imgviz.draw.line(img, yx=[(20, 20), (80, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(img, yx1=(20, 20), yx2=(80, 80), fill=(255, 0, 0))
    ys, xs = np.where(np.any(line != arrow, axis=2))
    assert ys.mean() > 50 and xs.mean() > 50  # head near tip (80, 80), not tail


def test_arrow_zero_length_skips_head() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    line = imgviz.draw.line(img, yx=[(50, 50), (50, 50)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(img, yx1=(50, 50), yx2=(50, 50), fill=(255, 0, 0))
    assert np.array_equal(arrow, line)  # no arrowhead, just the degenerate shaft


def test_arrow_in_place() -> None:
    pil = PIL.Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    before = np.asarray(pil).copy()
    imgviz.draw.arrow_(pil, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    assert not np.array_equal(np.asarray(pil), before)


def test_arrow_rejects_non_pil_image() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        imgviz.draw.arrow_(img, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))


@pytest.mark.parametrize("point", ["yx1", "yx2"])
def test_arrow_rejects_bad_point_shape(point: str) -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    kwargs = {"yx1": (20, 20), "yx2": (20, 80), point: (1, 2, 3)}
    with pytest.raises(ValueError, match="shape"):
        imgviz.draw.arrow(img, fill=(255, 0, 0), **kwargs)
