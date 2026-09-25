from typing import Literal

import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def rgb() -> NDArray[np.uint8]:
    return np.random.uniform(0, 255, size=(15, 25, 3)).round().astype(np.uint8)


def test_resize(rgb: NDArray[np.uint8]) -> None:
    dst = imgviz.resize(rgb, height=12)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8

    dst = imgviz.resize(rgb, width=20)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8

    dst = imgviz.resize(rgb, height=0.8)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8

    dst = imgviz.resize(rgb, width=0.8)
    assert dst.shape == (12, 20, 3)
    assert dst.dtype == np.uint8


@pytest.mark.parametrize(
    "height, width, expected_shape",
    [
        pytest.param(0.85, 0.75, (13, 19, 3), id="float-scale-both-axes"),
        pytest.param(None, 11, (7, 11, 3), id="width-only-derives-height"),
        pytest.param(13, None, (13, 22, 3), id="height-only-derives-width"),
    ],
)
def test_resize_rounds_fractional_dimension_to_nearest(
    rgb: NDArray[np.uint8],
    height: int | float | None,
    width: int | float | None,
    expected_shape: tuple[int, ...],
) -> None:
    dst = imgviz.resize(rgb, height=height, width=width)
    assert dst.shape == expected_shape


def test_resize_backends_agree_on_shape(rgb: NDArray[np.uint8]) -> None:
    via_pillow = imgviz.resize(rgb, height=12, backend="pillow")
    via_opencv = imgviz.resize(rgb, height=12, backend="opencv")

    assert via_pillow.shape == via_opencv.shape == (12, 20, 3)
    assert via_pillow.dtype == via_opencv.dtype == np.uint8


@pytest.mark.parametrize("backend", ["pillow", "opencv"])
@pytest.mark.parametrize("interpolation", ["linear", "nearest"])
def test_resize_preserves_solid_color(
    backend: Literal["pillow", "opencv"],
    interpolation: Literal["linear", "nearest"],
) -> None:
    solid = np.full((15, 25, 3), (10, 20, 30), dtype=np.uint8)

    dst = imgviz.resize(
        solid, height=12, width=20, interpolation=interpolation, backend=backend
    )

    assert dst.shape == (12, 20, 3)
    assert (dst == (10, 20, 30)).all()


@pytest.mark.parametrize("backend", ["pillow", "opencv"])
def test_resize_grayscale(
    rgb: NDArray[np.uint8], backend: Literal["pillow", "opencv"]
) -> None:
    gray = rgb[:, :, 0]

    dst = imgviz.resize(gray, height=12, width=20, backend=backend)

    assert dst.shape == (12, 20)
    assert dst.dtype == np.uint8


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resize_float_per_channel(dtype: type[np.floating]) -> None:
    img = np.random.rand(15, 25, 3).astype(dtype)

    dst = imgviz.resize(img, height=12, width=20, backend="pillow")

    assert dst.shape == (12, 20, 3)
    assert dst.dtype == dtype
    assert dst.min() != dst.max()


def test_resize_float_grayscale() -> None:
    img = np.random.rand(15, 25).astype(np.float32)

    dst = imgviz.resize(img, height=12, width=20, backend="pillow")

    assert dst.shape == (12, 20)
    assert dst.dtype == np.float32
    assert dst.min() != dst.max()


def test_resize_requires_height_or_width(rgb: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="either height or width must be given"):
        imgviz.resize(rgb)


def test_resize_rejects_unknown_backend(rgb: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="unsupported backend"):
        imgviz.resize(rgb, height=12, backend="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("backend", ["pillow", "opencv"])
def test_resize_rejects_unknown_interpolation(
    rgb: NDArray[np.uint8], backend: Literal["pillow", "opencv"]
) -> None:
    with pytest.raises(ValueError, match="unsupported interpolation"):
        imgviz.resize(
            rgb,
            height=12,
            interpolation="cubic",  # type: ignore[arg-type]
            backend=backend,
        )


def test_resize_rejects_non_array() -> None:
    with pytest.raises(TypeError, match="image type must be numpy.ndarray"):
        imgviz.resize([[1, 2], [3, 4]], height=4)  # type: ignore[arg-type]


def test_resize_rejects_non_numeric_dtype() -> None:
    with pytest.raises(TypeError, match="must be integer or floating"):
        imgviz.resize(np.zeros((4, 4), dtype=bool), height=2, backend="pillow")
