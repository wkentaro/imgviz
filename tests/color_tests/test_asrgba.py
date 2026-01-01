import numpy as np

import imgviz


def test_asrgba():
    data = imgviz.data.arc2017()
    gray = imgviz.rgb2gray(data["rgb"])

    rgba = imgviz.asrgba(gray)

    assert rgba.dtype == np.uint8
    assert rgba.ndim == 3
    assert rgba.shape[2] == 4
