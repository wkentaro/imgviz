import numpy as np

import imgviz


def test_depth2rgb():
    data = imgviz.data.arc2017()

    depthviz = imgviz.depth2rgb(data["depth"])

    assert depthviz.dtype == np.uint8
    H, W = data["depth"].shape[:2]
    assert depthviz.shape == (H, W, 3)
