import numpy as np

import imgviz


def test_asgray():
    data = imgviz.data.arc2017()
    gray = imgviz.asgray(data["rgb"])

    assert gray.ndim == 2
    assert gray.dtype == np.uint8
