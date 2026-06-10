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


def test_flow2rgb_handles_negative_zero_v() -> None:
    # A v-component of IEEE negative zero makes arctan2 return +pi, which
    # pushed the color-wheel index out of bounds. The vector (1, -0.0) is
    # identical to (1, +0.0), so it must produce the same visualization.
    flow_neg = np.zeros((4, 4, 2), dtype=np.float32)
    flow_neg[:, :, 0] = 1.0
    flow_neg[:, :, 1] = np.float32(-0.0)

    flow_pos = np.zeros((4, 4, 2), dtype=np.float32)
    flow_pos[:, :, 0] = 1.0

    flow_viz = imgviz.flow2rgb(flow_neg)

    assert flow_viz.dtype == np.uint8
    assert flow_viz.shape == (4, 4, 3)
    np.testing.assert_array_equal(flow_viz, imgviz.flow2rgb(flow_pos))
