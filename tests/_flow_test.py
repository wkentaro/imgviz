import numpy as np
from numpy.typing import NDArray

import imgviz


def test_flow2rgb():
    data = imgviz.data.middlebury()

    flow: NDArray[np.float32] = data["flow"]
    flowviz = imgviz.flow2rgb(flow)

    assert flowviz.dtype == np.uint8
    H, W = flow.shape[:2]
    assert flowviz.shape == (H, W, 3)
