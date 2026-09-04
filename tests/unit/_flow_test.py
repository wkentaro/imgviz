import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def flow_3_4() -> NDArray[np.float32]:
    return np.full((4, 4, 2), [3.0, 4.0], dtype=np.float32)


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


@pytest.mark.parametrize(
    ("uv", "expected_rgb"),
    [
        pytest.param((1.0, 0.0), (255, 17, 0), id="+u"),
        pytest.param((-1.0, 0.0), (0, 186, 255), id="-u"),
        pytest.param((0.0, 1.0), (255, 246, 0), id="+v"),
        pytest.param((0.0, -1.0), (107, 0, 255), id="-v"),
    ],
)
def test_flow2rgb_maps_cardinal_directions_to_colors(
    uv: tuple[float, float], expected_rgb: tuple[int, int, int]
) -> None:
    # The (u, v) direction selects the color-wheel hue; a u/v swap or an
    # arctan2 sign error would recolor every vector while passing shape and
    # dtype checks, so pin the four cardinal unit vectors to known colors.
    flow = np.full((2, 2, 2), uv, dtype=np.float32)

    flow_viz = imgviz.flow2rgb(flow, max_norm=1.0)

    np.testing.assert_array_equal(flow_viz[0, 0], expected_rgb)


def test_flow2rgb_rejects_non_3d() -> None:
    flow = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="flow must be 3 dimensional"):
        imgviz.flow2rgb(flow)


def test_flow2rgb_rejects_wrong_channel_count() -> None:
    flow = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=r"flow must have shape \(H, W, 2\)"):
        imgviz.flow2rgb(flow)


def test_flow2rgb_rejects_non_float_dtype() -> None:
    flow = np.zeros((4, 4, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="flow dtype must be float"):
        imgviz.flow2rgb(flow)


def test_flow2rgb_return_max_reports_norm(flow_3_4: NDArray[np.float32]) -> None:
    viz, max_norm = imgviz.flow2rgb(flow_3_4, return_max=True)

    assert max_norm == pytest.approx(5.0)
    np.testing.assert_array_equal(viz, imgviz.flow2rgb(flow_3_4, max_norm=max_norm))


def test_Flow2Rgb_caches_max_norm_after_first_call(
    flow_3_4: NDArray[np.float32],
) -> None:
    converter = imgviz.Flow2Rgb()
    assert converter.max_norm is None

    converter(flow_3_4)
    assert converter.max_norm == pytest.approx(5.0)

    flow_b = np.full((4, 4, 2), [6.0, 8.0], dtype=np.float32)
    viz_b = converter(flow_b)

    assert converter.max_norm == pytest.approx(5.0)
    np.testing.assert_array_equal(viz_b, imgviz.flow2rgb(flow_b, max_norm=5.0))


def test_Flow2Rgb_keeps_explicit_max_norm(flow_3_4: NDArray[np.float32]) -> None:
    converter = imgviz.Flow2Rgb(max_norm=2.0)
    assert converter.max_norm == pytest.approx(2.0)

    viz = converter(flow_3_4)

    assert converter.max_norm == pytest.approx(2.0)
    np.testing.assert_array_equal(viz, imgviz.flow2rgb(flow_3_4, max_norm=2.0))
