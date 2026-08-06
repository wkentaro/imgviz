import numpy as np
import pytest

import imgviz


@pytest.mark.parametrize(
    "shape",
    [(40, 50, 3), (40, 50), (40, 50, 2), (40, 50, 4), (40, 50, 5)],
    ids=["rgb", "gray", "la", "rgba", "five-channel"],
)
def test_blur_whole_image(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 256, size=shape).astype(np.uint8)

    dst = imgviz.blur(img, sigma=4.0)

    assert dst.shape == img.shape
    assert dst.dtype == img.dtype
    assert not np.array_equal(dst, img)


def test_blur_within_mask() -> None:
    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 256, size=(40, 50, 3)).astype(np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 15:35] = True

    dst = imgviz.blur(img, sigma=4.0, mask=mask)

    np.testing.assert_array_equal(dst[~mask], img[~mask])
    assert not np.array_equal(dst[mask], img[mask])


def test_blur_rejects_non_uint8_image() -> None:
    img = np.zeros((40, 50, 3), dtype=np.uint16)

    with pytest.raises(ValueError, match=r"image\.dtype must be uint8"):
        imgviz.blur(img, sigma=4.0)


def test_blur_rejects_negative_sigma() -> None:
    img = np.zeros((40, 50, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="sigma must be >= 0"):
        imgviz.blur(img, sigma=-1.0)


def test_blur_rejects_invalid_ndim() -> None:
    img = np.zeros((40, 50, 3, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"image\.ndim must be 2 or 3"):
        imgviz.blur(img, sigma=4.0)
