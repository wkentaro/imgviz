import numpy as np

import imgviz


def test_label2rgb():
    data = imgviz.data.kitti()
    img, flow = data['flow']

    flow_rgb = imgviz.flow2rgb(flow)

    assert flow_rgb.dtype == np.uint8
    H, W = flow.shape[:2]
    assert flow_rgb.shape == (H, W, 3)
