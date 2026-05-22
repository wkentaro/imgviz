import numpy as np
import pytest

from imgviz import _utils


def test_apply_mask_none_returns_transformed() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    transformed = np.ones((40, 50, 3), dtype=np.uint8)

    dst = _utils.apply_mask(image=image, transformed=transformed, mask=None)

    assert dst is transformed


def test_apply_mask_composites_within_mask() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    transformed = np.ones((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 15:35] = True

    dst = _utils.apply_mask(image=image, transformed=transformed, mask=mask)

    np.testing.assert_array_equal(dst[mask], transformed[mask])
    np.testing.assert_array_equal(dst[~mask], image[~mask])


def test_apply_mask_rejects_non_bool_mask() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask.dtype must be bool"):
        _utils.apply_mask(image=image, transformed=image, mask=mask)


def test_apply_mask_rejects_shape_mismatch() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="mask.shape must be"):
        _utils.apply_mask(image=image, transformed=image, mask=mask)
