import numpy as np

import imgviz


def test_asrgb():
    data = imgviz.data.arc2017()
    gray = imgviz.rgb2gray(data["rgb"])

    rgb = imgviz.asrgb(gray)

    assert rgb.dtype == np.uint8
    assert rgb.ndim == 3
