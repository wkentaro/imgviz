import numpy as np
import pytest

import imgviz


@pytest.mark.parametrize("dtype", [np.uint8, np.float32, np.float64])
def test_colorize(dtype: type[np.uint8] | type[np.floating]) -> None:
    data = imgviz.data.arc2017()

    out = imgviz.colorize(data["depth"], dtype=dtype)

    assert out.dtype == dtype
    H, W = data["depth"].shape[:2]
    assert out.shape == (H, W, 3)


def test_colorize_invalid_dtype() -> None:
    data = imgviz.data.arc2017()

    with pytest.raises(ValueError, match="dtype must be"):
        imgviz.colorize(data["depth"], dtype=np.int32)  # type: ignore[call-overload]


def test_colorize_default_cmap_is_viridis() -> None:
    data = imgviz.data.arc2017()

    viridis = imgviz.colorize(data["depth"])
    explicit = imgviz.colorize(data["depth"], cmap="viridis")
    jet = imgviz.colorize(data["depth"], cmap="jet")

    np.testing.assert_array_equal(viridis, explicit)
    assert not np.array_equal(viridis, jet)


def test_colorize_vmin_vmax() -> None:
    scalar = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)

    clipped = imgviz.colorize(scalar, vmin=0.25, vmax=0.75)

    assert clipped.shape == (10, 10, 3)
    assert clipped.dtype == np.uint8


def test_Colorize_caches_vmin_vmax_after_first_call() -> None:
    colorizer = imgviz.Colorize()
    first = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    second = np.linspace(0, 4, 100, dtype=np.float32).reshape(10, 10)

    assert colorizer.vmin is None
    assert colorizer.vmax is None

    colorizer(first)

    assert colorizer.vmin == pytest.approx(0.0)
    assert colorizer.vmax == pytest.approx(1.0)

    colorizer(second)

    assert colorizer.vmin == pytest.approx(0.0)
    assert colorizer.vmax == pytest.approx(1.0)
