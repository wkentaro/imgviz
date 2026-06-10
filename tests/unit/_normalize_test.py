import numpy as np
import pytest

import imgviz


def test_normalize() -> None:
    src = np.array([[0.0, 50.0], [100.0, 25.0]], dtype=np.float32)
    result = imgviz.normalize(src)
    assert result.shape == src.shape
    assert result.min() == 0
    assert result.max() == 1


def test_normalize_return_minmax() -> None:
    src = np.array([[0.0, 50.0], [100.0, 25.0]], dtype=np.float32)
    result, min_val, max_val = imgviz.normalize(src, return_minmax=True)
    assert result.shape == src.shape
    assert min_val.shape == (1,)
    assert max_val.shape == (1,)
    assert min_val[0] == 0.0
    assert max_val[0] == 100.0


def test_normalize_constant_large_float32_is_finite() -> None:
    # When min == max and the value is large enough that float32 eps rounds
    # away, the issame spread must remain large enough to survive the cast.
    src = np.full((4, 4), 1e6, dtype=np.float32)

    result = imgviz.normalize(src)

    assert result.shape == src.shape
    assert np.isfinite(result).all()


def test_normalize_multichannel_maps_each_channel_to_0_1() -> None:
    src = np.array(
        [
            [[0.0, 5.0, -2.0], [10.0, 15.0, 2.0]],
            [[5.0, 8.0, 0.0], [2.0, 12.0, 1.0]],
        ],
        dtype=np.float32,
    )

    result = imgviz.normalize(src)

    assert result.shape == src.shape
    np.testing.assert_allclose(result.min(axis=(0, 1)), [0.0, 0.0, 0.0])
    np.testing.assert_allclose(result.max(axis=(0, 1)), [1.0, 1.0, 1.0])


def test_normalize_multichannel_explicit_minmax() -> None:
    src = np.array(
        [[[0.0, 5.0, -2.0], [10.0, 15.0, 2.0]]],
        dtype=np.float32,
    )

    # Ranges wider than the data so the explicit override changes the output;
    # auto min/max would instead map the data to [0, 1].
    result, min_val, max_val = imgviz.normalize(
        src,
        min_value=[-10.0, 0.0, -4.0],
        max_value=[10.0, 20.0, 4.0],
        return_minmax=True,
    )

    assert result.shape == src.shape
    np.testing.assert_allclose(min_val, [-10.0, 0.0, -4.0])
    np.testing.assert_allclose(max_val, [10.0, 20.0, 4.0])
    np.testing.assert_allclose(result[0, 0], [0.5, 0.25, 0.25])
    np.testing.assert_allclose(result[0, 1], [1.0, 0.75, 0.75])


def test_normalize_nan_2d_stays_nan() -> None:
    src = np.array([[0.0, np.nan], [100.0, 50.0]], dtype=np.float32)

    result = imgviz.normalize(src)

    assert np.isnan(result[0, 1])
    assert result[0, 0] == 0.0
    assert result[1, 0] == 1.0


def test_normalize_nan_multichannel_marks_whole_pixel() -> None:
    src = np.tile(np.array([1.0, 2.0, 3.0], dtype=np.float32), (2, 2, 1))
    src[1, 1, 0] = np.nan

    result = imgviz.normalize(src)

    assert np.isnan(result[1, 1]).all()
    assert np.isfinite(result[0, 0]).all()


def test_normalize_rejects_bad_ndim() -> None:
    src = np.zeros((2, 2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="image ndim must be 2 or 3"):
        imgviz.normalize(src)


@pytest.mark.parametrize(
    ("min_value", "max_value", "match"),
    [
        ([0.0, 0.0], None, r"min_value\.shape must be"),
        (None, [1.0, 1.0], r"max_value\.shape must be"),
    ],
)
def test_normalize_rejects_wrong_value_shape(
    min_value: list[float] | None, max_value: list[float] | None, match: str
) -> None:
    src = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=match):
        imgviz.normalize(src, min_value=min_value, max_value=max_value)


def test_normalize_warns_on_inf() -> None:
    src = np.array([[0.0, 50.0], [100.0, 25.0]], dtype=np.float32)
    with pytest.warns(UserWarning, match=r"some of min or max values are inf\."):
        imgviz.normalize(src, min_value=0.0, max_value=np.inf)
