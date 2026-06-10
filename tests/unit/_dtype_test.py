import numpy as np
import pytest

import imgviz


def test_bool2ubyte() -> None:
    img = np.array([[True, False], [False, True]], dtype=bool)
    result = imgviz.bool2ubyte(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape
    assert result[0, 0] == 255
    assert result[0, 1] == 0


def test_float2ubyte() -> None:
    img = np.array([[0.0, 0.5], [1.0, 0.25]], dtype=np.float32)
    result = imgviz.float2ubyte(img)
    assert result.dtype == np.uint8
    assert result.shape == img.shape
    np.testing.assert_allclose(result, [[0, 128], [255, 64]])


def test_bool2ubyte_rejects_non_bool() -> None:
    with pytest.raises(ValueError, match="image dtype must be bool"):
        imgviz.bool2ubyte(np.zeros((2, 2), dtype=np.uint8))


def test_float2ubyte_rejects_non_float() -> None:
    with pytest.raises(ValueError, match="image dtype must be float"):
        imgviz.float2ubyte(np.zeros((2, 2), dtype=np.uint8))


def test_float2ubyte_rejects_below_zero() -> None:
    with pytest.raises(ValueError, match=r"image\.min\(\) must be >= 0"):
        imgviz.float2ubyte(np.array([-0.1, 0.5], dtype=np.float32))


def test_float2ubyte_rejects_above_one() -> None:
    with pytest.raises(ValueError, match=r"image\.max\(\) must be <= 1"):
        imgviz.float2ubyte(np.array([0.5, 1.1], dtype=np.float32))
