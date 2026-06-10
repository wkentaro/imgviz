import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_box_corners(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.box_corners(
        white_image, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0)
    )
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert not np.array_equal(res, white_image)


def test_box_corners_four_corners_symmetric() -> None:
    img = np.full((101, 101, 3), 255, dtype=np.uint8)
    res = imgviz.draw.box_corners(
        img, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0), length=15, width=1
    )
    changed = np.any(res != img, axis=2)  # box centered on (50, 50)
    assert np.array_equal(changed, np.fliplr(changed))
    assert np.array_equal(changed, np.flipud(changed))


def test_box_corners_draws_only_corners(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.box_corners(
        white_image, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0), length=12
    )
    assert np.all(res[20, 40:61] == 255)  # middle of the top edge stays blank


def test_box_corners_degrades_to_rectangle_when_length_exceeds_box(
    white_image: NDArray[np.uint8],
) -> None:
    res = imgviz.draw.box_corners(
        white_image, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0), length=1000
    )
    assert np.array_equal(res[20, 50], [0, 255, 0])  # top edge now fully drawn


def test_box_corners_respects_width(white_image: NDArray[np.uint8]) -> None:
    thin = imgviz.draw.box_corners(
        white_image, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0), width=1
    )
    thick = imgviz.draw.box_corners(
        white_image, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0), width=6
    )
    n_thin = np.any(thin != white_image, axis=2).sum()
    n_thick = np.any(thick != white_image, axis=2).sum()
    assert n_thin < n_thick


def test_box_corners_rejects_inverted_box(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="min vertex"):
        imgviz.draw.box_corners(
            white_image, yx1=(80, 80), yx2=(20, 20), fill=(0, 255, 0)
        )


def test_box_corners_in_place() -> None:
    pil = PIL.Image.fromarray(np.full((100, 100, 3), 255, dtype=np.uint8))
    before = np.asarray(pil).copy()
    imgviz.draw.box_corners_(pil, yx1=(20, 20), yx2=(80, 80), fill=(0, 255, 0))
    assert not np.array_equal(np.asarray(pil), before)
