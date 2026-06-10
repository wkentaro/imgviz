import numpy as np
import pytest
from numpy.typing import ArrayLike

import imgviz


def test_heatmap() -> None:
    res = imgviz.heatmap([(50, 50)], shape=(100, 100), sigma=10)
    assert res.shape == (100, 100)
    assert res.dtype == np.float64
    assert np.unravel_index(res.argmax(), res.shape) == (50, 50)


def test_heatmap_empty_is_zero() -> None:
    res = imgviz.heatmap([], shape=(100, 100))
    assert res.shape == (100, 100)
    np.testing.assert_array_equal(res, np.zeros((100, 100)))


def test_heatmap_peak_equals_weight_and_falls_off() -> None:
    res = imgviz.heatmap([(50, 50)], shape=(100, 100), sigma=10)
    assert res[50, 50] == pytest.approx(1.0)
    assert res[50, 60] == pytest.approx(np.exp(-0.5))  # value one sigma away
    assert res[50, 50] > res[50, 60] > res[50, 70]


def test_heatmap_sigma_controls_spread() -> None:
    narrow = imgviz.heatmap([(50, 50)], shape=(100, 100), sigma=5)
    wide = imgviz.heatmap([(50, 50)], shape=(100, 100), sigma=20)
    assert wide[50, 70] > narrow[50, 70]  # wider Gaussian reaches further


def test_heatmap_weights() -> None:
    res = imgviz.heatmap(
        [(30, 30), (70, 70)], shape=(100, 100), sigma=8, weights=[1.0, 3.0]
    )
    assert res[70, 70] == pytest.approx(3.0 * res[30, 30])


def test_heatmap_sums_overlapping_points() -> None:
    one = imgviz.heatmap([(50, 50)], shape=(100, 100), sigma=10)
    two = imgviz.heatmap([(50, 50), (50, 50)], shape=(100, 100), sigma=10)
    assert two[50, 50] == pytest.approx(2.0 * one[50, 50])


def test_heatmap_composes_with_colorize() -> None:
    density = imgviz.heatmap([(50, 50), (20, 80)], shape=(100, 100))
    viz = imgviz.colorize(density)
    assert viz.shape == (100, 100, 3)
    assert viz.dtype == np.uint8


@pytest.mark.parametrize("points", [[[1, 2, 3]], np.zeros((0, 3))])
def test_heatmap_rejects_bad_points_shape(points: ArrayLike) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        imgviz.heatmap(points, shape=(10, 10))


def test_heatmap_rejects_bad_weights_shape() -> None:
    with pytest.raises(ValueError, match="weights must have shape"):
        imgviz.heatmap([(1, 1), (2, 2)], shape=(10, 10), weights=[1.0])


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf")])
def test_heatmap_rejects_invalid_sigma(sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma must be"):
        imgviz.heatmap([(1, 1)], shape=(10, 10), sigma=sigma)
