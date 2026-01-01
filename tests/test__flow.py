import numpy as np

import imgviz


def test_flow2rgb():
    data = imgviz.data.middlebury()

    flow = data["flow"]
    flowviz = imgviz.flow2rgb(flow)

    assert flowviz.dtype == np.uint8
    H, W = flow.shape[:2]
    assert flowviz.shape == (H, W, 3)
