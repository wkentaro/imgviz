import numpy as np

import imgviz


def test_normalize() -> None:
    src = np.array([[0.0, 50.0], [100.0, 25.0]], dtype=np.float32)
    result = imgviz.normalize(src)
    assert result.shape == src.shape
    assert result.min() == 0
    assert result.max() == 1


def test_normalize_return_minmax() -> None:
    src = np.array([[0.0, 50.0], [100.0, 25.0]], dtype=np.float32)
    result, min_val, max_val = imgviz.normalize(src, return_minmax=True)
    assert result.shape == src.shape
    assert min_val.shape == (1,)
    assert max_val.shape == (1,)
    assert min_val[0] == 0.0
    assert max_val[0] == 100.0
