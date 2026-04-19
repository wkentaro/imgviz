import numpy as np
import pytest

import imgviz


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_depth2rgb(dtype: type[np.uint8] | type[np.floating]) -> None:
    data = imgviz.data.arc2017()

    with pytest.warns(DeprecationWarning, match="depth2rgb"):
        depthviz = imgviz.depth2rgb(data["depth"], dtype=dtype)

    assert depthviz.dtype == dtype
    H, W = data["depth"].shape[:2]
    assert depthviz.shape == (H, W, 3)


def test_depth2rgb_invalid_dtype() -> None:
    data = imgviz.data.arc2017()

    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="dtype must be"):
            imgviz.depth2rgb(data["depth"], dtype=np.int32)  # type: ignore[call-overload]
