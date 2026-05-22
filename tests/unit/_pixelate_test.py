import numpy as np

import imgviz


def test_pixelate_whole_image() -> None:
    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 256, size=(40, 50, 3)).astype(np.uint8)

    dst = imgviz.pixelate(img, block=4)

    assert dst.shape == img.shape
    assert dst.dtype == img.dtype
    assert not np.array_equal(dst, img)


def test_pixelate_within_mask() -> None:
    rng = np.random.default_rng(seed=0)
    img = rng.integers(0, 256, size=(40, 50, 3)).astype(np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 15:35] = True

    dst = imgviz.pixelate(img, block=4, mask=mask)

    np.testing.assert_array_equal(dst[~mask], img[~mask])
    assert not np.array_equal(dst[mask], img[mask])


def test_pixelate_block_one_is_noop() -> None:
    rng = np.random.default_rng(seed=1)
    img = rng.integers(0, 256, size=(20, 20, 3)).astype(np.uint8)

    dst = imgviz.pixelate(img, block=1)

    np.testing.assert_array_equal(dst, img)
