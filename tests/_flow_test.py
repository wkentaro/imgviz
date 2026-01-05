import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.mark.parametrize(
    "use_class", [pytest.param(False, id="flow2rgb"), pytest.param(True, id="Flow2Rgb")]
)
def test_flow2rgb(use_class: bool, show: bool) -> None:
    data = imgviz.data.middlebury()

    flow: NDArray[np.float32] = data["flow"]

    flow_viz: NDArray[np.uint8]
    if use_class:
        flow_viz = imgviz.Flow2Rgb()(flow)
    else:
        flow_viz = imgviz.flow2rgb(flow)
    if show:
        plt.imshow(flow_viz)
        plt.show()

    assert flow_viz.dtype == np.uint8
    H, W = flow.shape[:2]
    assert flow_viz.shape == (H, W, 3)
