import numpy as np
import PIL.Image
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((100, 100, 3), 255, dtype=np.uint8)


def test_polygon_outline_only(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.polygon(
        white_image, yx=[(10, 10), (10, 90), (90, 90), (90, 10)], outline=(0, 0, 0)
    )
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    top_edge = res[10, 40:60]
    assert ((top_edge == (0, 0, 0)).all(axis=-1)).any()


def test_polygon_fill_only(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.polygon(
        white_image, yx=[(10, 10), (10, 90), (90, 50)], fill=(255, 0, 0)
    )
    assert (res[50, 50] == (255, 0, 0)).all()


def test_polygon_fill_and_outline_with_width(
    white_image: NDArray[np.uint8],
) -> None:
    res = imgviz.draw.polygon(
        white_image,
        yx=[(10, 10), (10, 90), (90, 90), (90, 10)],
        fill=(0, 255, 0),
        outline=(0, 0, 255),
        width=3,
    )
    assert (res[50, 50] == (0, 255, 0)).all()
    top_edge = res[10, 40:60]
    assert ((top_edge == (0, 0, 255)).all(axis=-1)).any()


def test_polygon_accepts_ndarray_yx(white_image: NDArray[np.uint8]) -> None:
    yx = np.array([[10, 10], [10, 90], [90, 50]], dtype=np.float32)
    res = imgviz.draw.polygon(white_image, yx=yx, fill=(0, 0, 0))
    assert not np.array_equal(res, white_image)


def test_polygon_rejects_non_2d_yx(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="yx must be 2D"):
        imgviz.draw.polygon(white_image, yx=[10, 10, 10, 90], fill=(0, 0, 0))


def test_polygon_rejects_wrong_inner_dim(
    white_image: NDArray[np.uint8],
) -> None:
    with pytest.raises(ValueError, match=r"yx.shape\[1\] must be 2"):
        imgviz.draw.polygon(white_image, yx=[(10, 10, 0), (10, 90, 0)], fill=(0, 0, 0))


def test_polygon_underscore_rejects_numpy_image(
    white_image: NDArray[np.uint8],
) -> None:
    with pytest.raises(TypeError, match="image must be PIL.Image.Image"):
        imgviz.draw.polygon_(
            white_image,  # type: ignore[arg-type]
            yx=[(10, 10), (10, 90), (90, 50)],
            fill=(0, 0, 0),
        )


def test_polygon_underscore_mutates_in_place(
    white_image: NDArray[np.uint8],
) -> None:
    pil = PIL.Image.fromarray(white_image)
    imgviz.draw.polygon_(pil, yx=[(10, 10), (10, 90), (90, 50)], fill=(0, 0, 0))
    assert not np.array_equal(np.array(pil), white_image)


def test_polygon_rejects_missing_fill_and_outline(
    white_image: NDArray[np.uint8],
) -> None:
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        imgviz.draw.polygon(white_image, yx=[(10, 10), (10, 90), (90, 50)])


def test_polygon_rejects_too_few_vertices(white_image: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="at least 3 vertices"):
        imgviz.draw.polygon(white_image, yx=[(10, 10), (10, 90)], fill=(0, 0, 0))
