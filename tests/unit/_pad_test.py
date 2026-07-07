import numpy as np
import pytest

import imgviz


def test_pad_hw_grayscale() -> None:
    image = np.random.randint(0, 256, size=(10, 20), dtype=np.uint8)

    dst = imgviz.pad(image, top=2, bottom=3, left=4, right=5, color=0)

    assert dst.shape == (15, 29)
    assert dst.dtype == image.dtype
    np.testing.assert_array_equal(dst[2:12, 4:24], image)
    assert (dst[:2] == 0).all()
    assert (dst[-3:] == 0).all()
    assert (dst[:, :4] == 0).all()
    assert (dst[:, -5:] == 0).all()


def test_pad_hwc_rgb() -> None:
    image = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)

    dst = imgviz.pad(image, top=1, bottom=1, left=1, right=1, color=(10, 20, 30))

    assert dst.shape == (10, 10, 3)
    np.testing.assert_array_equal(dst[1:9, 1:9], image)
    np.testing.assert_array_equal(dst[0, 0], [10, 20, 30])
    np.testing.assert_array_equal(dst[-1, -1], [10, 20, 30])


def test_pad_hwca_rgba() -> None:
    image = np.random.randint(0, 256, size=(8, 8, 4), dtype=np.uint8)

    dst = imgviz.pad(image, top=2, color=(0, 0, 0, 255))

    assert dst.shape == (10, 8, 4)
    np.testing.assert_array_equal(dst[2:], image)
    assert (dst[:2, :, 3] == 255).all()


def test_pad_directional_only_one_side() -> None:
    image = np.full((4, 4, 3), 7, dtype=np.uint8)

    dst = imgviz.pad(image, right=3, color=0)

    assert dst.shape == (4, 7, 3)
    np.testing.assert_array_equal(dst[:, :4], image)
    assert (dst[:, 4:] == 0).all()


def test_pad_scalar_color_fills_every_channel() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    dst = imgviz.pad(image, top=1, color=255)

    np.testing.assert_array_equal(dst[0], np.full((4, 3), 255, dtype=np.uint8))


def test_pad_color_ndarray() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    dst = imgviz.pad(image, left=1, color=np.array([1, 2, 3], dtype=np.uint8))

    np.testing.assert_array_equal(dst[:, 0], np.tile([1, 2, 3], (4, 1)))


def test_pad_zero_on_all_sides_returns_equal_array() -> None:
    image = np.random.randint(0, 256, size=(5, 6, 3), dtype=np.uint8)

    dst = imgviz.pad(image)

    assert dst is not image
    np.testing.assert_array_equal(dst, image)


def test_pad_preserves_float_dtype() -> None:
    image = np.random.rand(4, 4).astype(np.float32)

    dst = imgviz.pad(image, top=1, color=0.5)

    assert dst.dtype == np.float32
    assert (dst[0] == np.float32(0.5)).all()


def test_pad_negative_border_raises() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="top must be >= 0"):
        imgviz.pad(image, top=-1)


def test_pad_non_int_border_raises() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(TypeError, match="top must be int, but got float"):
        imgviz.pad(image, top=1.5)  # ty: ignore[invalid-argument-type]


def test_pad_invalid_ndim_raises() -> None:
    image = np.zeros((4,), dtype=np.uint8)

    with pytest.raises(ValueError, match="image.ndim must be 2 or 3"):
        imgviz.pad(image, top=1)


def test_pad_tuple_color_on_grayscale_raises() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="single-channel"):
        imgviz.pad(image, top=1, color=(0, 0, 0))


def test_pad_color_channel_mismatch_raises() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="color has 4 components but image has 3"):
        imgviz.pad(image, top=1, color=(0, 0, 0, 255))
