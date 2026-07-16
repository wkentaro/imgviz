import numpy as np
import pytest

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


def test_pixelate_blends_each_block_instead_of_sampling_a_corner() -> None:
    # Each 2x2 block holds only its two extreme values, so blending them
    # (downsample) lands strictly inside the block's range while sampling one
    # corner (nearest) would return an extreme. Left block spans 0..200, right
    # 40..120. The exact blended value is resize-backend dependent, so assert
    # the range rather than a literal.
    gray = np.array([[0, 200, 40, 120], [200, 0, 120, 40]], dtype=np.uint8)
    img = gray[:, :, None].repeat(3, axis=2)

    dst = imgviz.pixelate(img, block=2)[:, :, 0]

    assert 0 < dst[0, 0] < 200
    assert 40 < dst[0, 2] < 120


@pytest.mark.parametrize("block", [0, -1])
def test_pixelate_rejects_block_below_one(block: int) -> None:
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="block must be >= 1"):
        imgviz.pixelate(img, block=block)


def test_pixelate_block_exceeds_both_dimensions_yields_uniform() -> None:
    rng = np.random.default_rng(seed=2)
    img = rng.integers(0, 256, size=(8, 8, 3)).astype(np.uint8)

    dst = imgviz.pixelate(img, block=100)

    assert dst.shape == img.shape
    np.testing.assert_array_equal(dst, np.full_like(dst, dst[0, 0]))
