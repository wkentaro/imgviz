import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_arrow(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.arrow(white_image, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert not np.array_equal(res, white_image)


def test_arrow_adds_head_over_plain_line(white_image: NDArray[np.uint8]) -> None:
    line = imgviz.draw.line(white_image, yx=[(20, 20), (20, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(white_image, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    changed_by_head = np.any(line != arrow, axis=2)
    assert changed_by_head.sum() > 0


def test_arrow_head_is_at_the_tip(white_image: NDArray[np.uint8]) -> None:
    line = imgviz.draw.line(white_image, yx=[(20, 20), (20, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(white_image, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    ys, xs = np.where(np.any(line != arrow, axis=2))
    assert xs.mean() > 50  # closer to the tip (x=80) than the tail (x=20)


def test_arrow_head_length_ratio_scales_head(white_image: NDArray[np.uint8]) -> None:
    small = imgviz.draw.arrow(
        white_image, yx1=(50, 10), yx2=(50, 90), fill=(255, 0, 0), head_length_ratio=0.1
    )
    large = imgviz.draw.arrow(
        white_image, yx1=(50, 10), yx2=(50, 90), fill=(255, 0, 0), head_length_ratio=0.3
    )
    small_changed = np.any(small != white_image, axis=2).sum()
    large_changed = np.any(large != white_image, axis=2).sum()
    assert small_changed < large_changed


def test_arrow_head_is_at_the_tip_diagonal(white_image: NDArray[np.uint8]) -> None:
    line = imgviz.draw.line(white_image, yx=[(20, 20), (80, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(white_image, yx1=(20, 20), yx2=(80, 80), fill=(255, 0, 0))
    ys, xs = np.where(np.any(line != arrow, axis=2))
    assert ys.mean() > 50 and xs.mean() > 50  # head near tip (80, 80), not tail


def test_arrow_head_is_symmetric_about_the_shaft(
    white_image: NDArray[np.uint8],
) -> None:
    line = imgviz.draw.line(white_image, yx=[(20, 20), (80, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(white_image, yx1=(20, 20), yx2=(80, 80), fill=(255, 0, 0))
    ys, xs = np.where(np.any(line != arrow, axis=2))
    head = set(zip(ys.tolist(), xs.tolist()))
    # the shaft lies on the y=x diagonal, so a symmetric head is invariant under
    # reflecting across it (swapping y and x)
    reflected = {(x, y) for y, x in head}
    assert head == reflected


def test_arrow_head_is_symmetric_about_a_horizontal_shaft(
    white_image: NDArray[np.uint8],
) -> None:
    line = imgviz.draw.line(white_image, yx=[(50, 20), (50, 80)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(white_image, yx1=(50, 20), yx2=(50, 80), fill=(255, 0, 0))
    ys, xs = np.where(np.any(line != arrow, axis=2))
    head = set(zip(ys.tolist(), xs.tolist()))
    # the shaft lies on row y=50, so a symmetric head is invariant under
    # reflecting across it (y -> 2 * 50 - y)
    reflected = {(2 * 50 - y, x) for y, x in head}
    assert head == reflected


def test_arrow_zero_length_skips_head(white_image: NDArray[np.uint8]) -> None:
    line = imgviz.draw.line(white_image, yx=[(50, 50), (50, 50)], fill=(255, 0, 0))
    arrow = imgviz.draw.arrow(white_image, yx1=(50, 50), yx2=(50, 50), fill=(255, 0, 0))
    assert np.array_equal(arrow, line)  # no arrowhead, just the degenerate shaft


def test_arrow_in_place(white_image: NDArray[np.uint8]) -> None:
    pil = PIL.Image.fromarray(white_image)
    before = np.asarray(pil).copy()
    imgviz.draw.arrow_(pil, yx1=(20, 20), yx2=(20, 80), fill=(255, 0, 0))
    assert not np.array_equal(np.asarray(pil), before)


def test_arrow_rejects_non_pil_image(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        imgviz.draw.arrow_(
            white_image,  # type: ignore[arg-type]
            yx1=(20, 20),
            yx2=(20, 80),
            fill=(255, 0, 0),
        )


@pytest.mark.parametrize("point", ["yx1", "yx2"])
def test_arrow_rejects_bad_point_shape(
    point: str, white_image: NDArray[np.uint8]
) -> None:
    kwargs = {"yx1": (20, 20), "yx2": (20, 80), point: (1, 2, 3)}
    with pytest.raises(ValueError, match="shape"):
        imgviz.draw.arrow(
            white_image,
            fill=(255, 0, 0),
            **kwargs,  # type: ignore[arg-type]
        )
