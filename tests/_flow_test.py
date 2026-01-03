import typing

import numpy as np
from numpy.typing import NDArray

import imgviz


def test_flow2rgb():
    data = imgviz.data.middlebury()

    assert isinstance(data["flow"], np.ndarray)
    assert data["flow"].dtype == np.float32
    flow: NDArray[np.float32] = typing.cast(NDArray[np.float32], data["flow"])
    flowviz = imgviz.flow2rgb(flow)

    assert flowviz.dtype == np.uint8
    H, W = flow.shape[:2]
    assert flowviz.shape == (H, W, 3)
