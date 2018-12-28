import numpy as np

import imgviz


def test_label2rgb():
    data = imgviz.data.arc2017()

    labelviz = imgviz.label2rgb(data['class_label'])

    assert labelviz.dtype == np.uint8
    H, W = data['class_label'].shape[:2]
    assert labelviz.shape == (H, W, 3)
