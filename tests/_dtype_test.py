import numpy as np

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
