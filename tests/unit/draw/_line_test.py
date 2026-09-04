import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_line(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.line(white_image, yx=[(10, 10), (90, 90)], fill=(0, 0, 0))
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert (res[50, 50] == (0, 0, 0)).all()  # line lands on the diagonal


def test_line_points_are_yx_not_xy(white_image: NDArray[np.uint8]) -> None:
    # Vertical segment at column x=80; a (y, x) -> (x, y) swap would instead
    # draw it horizontally at row y=80.
    res = imgviz.draw.line(white_image, yx=[(10, 80), (90, 80)], fill=(0, 0, 0))
    assert (res[50, 80] == (0, 0, 0)).all()  # on the vertical line
    assert (res[80, 50] == (255, 255, 255)).all()  # reflected point stays white


def test_line_underscore_mutates_in_place(white_image: NDArray[np.uint8]) -> None:
    image = PIL.Image.fromarray(white_image)

    imgviz.draw.line_(image, yx=[(10, 10), (90, 90)], fill=(0, 0, 0))

    assert not np.array_equal(np.asarray(image), white_image)


def test_line_accepts_ndarray_yx(white_image: NDArray[np.uint8]) -> None:
    yx = np.array([[10, 10], [90, 90]], dtype=np.float32)

    res = imgviz.draw.line(white_image, yx=yx, fill=(0, 0, 0))

    assert not np.array_equal(res, white_image)


def test_line_rejects_non_2d_yx(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="yx must be 2D"):
        imgviz.draw.line(white_image, yx=[10, 10, 90, 90], fill=(0, 0, 0))


def test_line_rejects_wrong_inner_dim(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match=r"yx\.shape\[1\] must be 2"):
        imgviz.draw.line(white_image, yx=[(10, 10, 0), (90, 90, 0)], fill=(0, 0, 0))


def test_line_underscore_rejects_numpy_image(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(TypeError, match="image must be PIL.Image.Image"):
        imgviz.draw.line_(
            white_image,  # type: ignore[arg-type]
            yx=[(10, 10), (90, 90)],
            fill=(0, 0, 0),
        )
