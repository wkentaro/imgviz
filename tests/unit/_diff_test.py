from typing import Literal

import cmap
import numpy as np
import pytest

import imgviz


def test_diff_signed_centered_at_zero() -> None:
    gradient = np.linspace(-1, 1, 9, dtype=np.float32)
    a = np.tile(gradient, (9, 1))
    b = np.zeros((9, 9), dtype=np.float32)

    out = imgviz.diff(a, b, mode="signed")

    assert out.shape == (9, 9, 3)
    assert out.dtype == np.uint8

    center = (
        (np.asarray(cmap.Colormap("coolwarm")(0.5))[:3] * 255).round().astype(np.uint8)
    )
    np.testing.assert_array_equal(out[:, 4], np.tile(center, (9, 1)))
    assert not np.array_equal(out[:, 0], out[:, -1])


@pytest.mark.parametrize(
    ("vmin", "vmax"),
    [(-2.0, None), (None, 2.0), (2.0, None), (None, -2.0)],
)
def test_diff_signed_single_bound_is_symmetric(
    vmin: float | None, vmax: float | None
) -> None:
    a = np.tile(np.linspace(-3, 3, 9, dtype=np.float32), (9, 1))
    b = np.zeros((9, 9), dtype=np.float32)

    out = imgviz.diff(a, b, mode="signed", vmin=vmin, vmax=vmax)

    reference = imgviz.diff(a, b, mode="signed", vmin=-2.0, vmax=2.0)
    np.testing.assert_array_equal(out, reference)


def test_diff_signed_symmetric_under_swap() -> None:
    a = np.tile(np.linspace(-1, 1, 9, dtype=np.float32), (9, 1))
    b = np.zeros((9, 9), dtype=np.float32)

    out_forward = imgviz.diff(a, b, mode="signed")
    out_swapped = imgviz.diff(b, a, mode="signed")

    np.testing.assert_array_equal(out_forward, out_swapped[:, ::-1])


@pytest.mark.parametrize("mode", ["signed", "abs", "ssim"])
def test_diff_identical_inputs_are_neutral(
    mode: Literal["signed", "abs", "ssim"],
) -> None:
    image = imgviz.data.arc2017()["rgb"]

    out = imgviz.diff(image, image.copy(), mode=mode)

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.tile(out[0, 0], (*image.shape[:2], 1)))


def test_diff_signed_identical_inputs_have_coolwarm_midpoint() -> None:
    image = imgviz.data.arc2017()["rgb"]

    out = imgviz.diff(image, image.copy(), mode="signed")

    midpoint = (
        (np.asarray(cmap.Colormap("coolwarm")(0.5))[:3] * 255).round().astype(np.uint8)
    )
    np.testing.assert_array_equal(out[0, 0], midpoint)


def test_diff_signed_asymmetric_bounds_are_used_as_is() -> None:
    a = np.tile(np.linspace(-2, 2, 9, dtype=np.float32), (9, 1))
    b = np.zeros((9, 9), dtype=np.float32)

    out = imgviz.diff(a, b, mode="signed", vmin=0.0, vmax=2.0)

    # vmin=0 saturates negative diffs to coolwarm(0); positive diffs span
    # coolwarm(0..1). The center column (diff=0) sits at the low end, not
    # the midpoint.
    low_end = (
        (np.asarray(cmap.Colormap("coolwarm")(0.0))[:3] * 255).round().astype(np.uint8)
    )
    high_end = (
        (np.asarray(cmap.Colormap("coolwarm")(1.0))[:3] * 255).round().astype(np.uint8)
    )
    np.testing.assert_array_equal(out[:, 4], np.tile(low_end, (9, 1)))
    np.testing.assert_array_equal(out[:, 0], np.tile(low_end, (9, 1)))
    np.testing.assert_array_equal(out[:, -1], np.tile(high_end, (9, 1)))


@pytest.mark.parametrize(("vmin", "vmax"), [(0.0, None), (None, 0.0), (0.0, 0.0)])
def test_diff_signed_zero_bound_falls_back_to_data(
    vmin: float | None, vmax: float | None
) -> None:
    a = np.tile(np.linspace(-2, 2, 9, dtype=np.float32), (9, 1))
    b = np.zeros((9, 9), dtype=np.float32)

    out = imgviz.diff(a, b, mode="signed", vmin=vmin, vmax=vmax)
    reference = imgviz.diff(a, b, mode="signed")

    np.testing.assert_array_equal(out, reference)

    midpoint = (
        (np.asarray(cmap.Colormap("coolwarm")(0.5))[:3] * 255).round().astype(np.uint8)
    )
    low_end = (
        (np.asarray(cmap.Colormap("coolwarm")(0.0))[:3] * 255).round().astype(np.uint8)
    )
    np.testing.assert_array_equal(out[:, 4], np.tile(midpoint, (9, 1)))
    np.testing.assert_array_equal(out[:, 0], np.tile(low_end, (9, 1)))


def test_diff_abs_grows_with_difference() -> None:
    a = np.zeros((9, 9), dtype=np.float32)
    small = np.full((9, 9), 0.2, dtype=np.float32)
    large = np.full((9, 9), 0.8, dtype=np.float32)

    out_small = imgviz.diff(a, small, mode="abs", vmax=1.0)
    out_large = imgviz.diff(a, large, mode="abs", vmax=1.0)

    assert not np.array_equal(out_small[0, 0], out_large[0, 0])


def test_diff_ssim_stable_on_near_uniform_uint8() -> None:
    # Regression: when data_range was derived from observed pixel values, a
    # tiny change on a uniform uint8 surface collapsed SSIM toward 0. With
    # data_range taken from the uint8 dtype, scores stay near 1 everywhere.
    a = np.full((32, 32), 128, dtype=np.uint8)
    b = a.copy()
    b[16, 16] = 130

    out = imgviz.diff(a, b, mode="ssim", vmin=-1.0, vmax=1.0)

    near_perfect = (
        (np.asarray(cmap.Colormap("viridis")(1.0))[:3] * 255).round().astype(np.uint8)
    )
    far_pixel = out[0, 0]
    assert np.all(np.abs(far_pixel.astype(int) - near_perfect.astype(int)) <= 2), (
        f"expected far pixel ~ {near_perfect}, got {far_pixel}"
    )


def test_diff_ssim_mixed_dtypes_uses_observed_range() -> None:
    # The dtype-derived data_range only applies when both inputs share an
    # integer dtype. Mixed inputs (e.g. uint8 against float32) must not
    # silently key on iinfo(a.dtype); they fall back to the observed extent.
    a = np.full((16, 16), 0.5, dtype=np.float32)
    b = np.full((16, 16), 128, dtype=np.uint8)

    out = imgviz.diff(a, b, mode="ssim", vmin=-1.0, vmax=1.0)

    assert out.shape == (16, 16, 3)
    assert out.dtype == np.uint8


def test_diff_ssim_highlights_local_change() -> None:
    image = imgviz.data.arc2017()["rgb"]
    modified = image.copy()
    modified[:50, :50] = 0

    out = imgviz.diff(image, modified, mode="ssim")

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, np.tile(out[-1, -1], (*image.shape[:2], 1)))


def test_diff_accepts_rgb_input() -> None:
    image = imgviz.data.arc2017()["rgb"]

    out = imgviz.diff(image, image.copy(), mode="abs")

    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_diff_accepts_rgba_input() -> None:
    rgb = imgviz.data.arc2017()["rgb"]
    a = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    b = a.copy()
    b[:50, :50, :3] = 0

    out = imgviz.diff(a, b, mode="abs")

    assert out.shape == (*a.shape[:2], 3)
    assert out.dtype == np.uint8
    assert not np.array_equal(out[0, 0], out[-1, -1])


def test_diff_rgba_alpha_is_ignored() -> None:
    rgb = imgviz.data.arc2017()["rgb"]
    a = np.dstack([rgb, np.full(rgb.shape[:2], 255, dtype=np.uint8)])
    b = a.copy()
    b[..., 3] = 0

    out = imgviz.diff(a, b, mode="abs")

    np.testing.assert_array_equal(out, np.tile(out[0, 0], (*a.shape[:2], 1)))


def test_diff_ssim_propagates_nan_locally() -> None:
    rng = np.random.RandomState(0)
    a = (100.0 * rng.rand(32, 32)).astype(np.float32)
    b = a + 5.0 * rng.rand(32, 32).astype(np.float32)
    a[0, 0] = np.nan

    out = imgviz.diff(a, b, mode="ssim", vmin=-1.0, vmax=1.0)

    assert out.shape == (32, 32, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], (0, 0, 0))
    assert not np.array_equal(out[-1, -1], (0, 0, 0))


@pytest.mark.parametrize("mode", ["signed", "abs"])
def test_diff_propagates_nan(mode: Literal["signed", "abs"]) -> None:
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)
    a[0, 0] = np.nan
    a[1, 1] = 1.0

    out = imgviz.diff(a, b, mode=mode)

    assert out.shape == (4, 4, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], (0, 0, 0))


def test_diff_shape_mismatch_raises() -> None:
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match="same shape"):
        imgviz.diff(a, b)


def test_diff_invalid_mode_raises() -> None:
    a = np.zeros((4, 4), dtype=np.float32)
    b = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="mode must be"):
        imgviz.diff(a, b, mode="bogus")  # type: ignore[arg-type]


def test_diff_ssim_small_image_raises() -> None:
    a = np.zeros((6, 100), dtype=np.float32)
    b = np.zeros((6, 100), dtype=np.float32)

    with pytest.raises(ValueError, match="at least 7"):
        imgviz.diff(a, b, mode="ssim")


@pytest.mark.parametrize(
    ("a_all_nan", "b_all_nan"),
    [(True, True), (True, False), (False, True)],
)
def test_diff_ssim_all_nan_inputs(a_all_nan: bool, b_all_nan: bool) -> None:
    shape = (16, 16)
    a = (
        np.full(shape, np.nan, dtype=np.float32)
        if a_all_nan
        else np.zeros(shape, dtype=np.float32)
    )
    b = (
        np.full(shape, np.nan, dtype=np.float32)
        if b_all_nan
        else np.zeros(shape, dtype=np.float32)
    )

    out = imgviz.diff(a, b, mode="ssim")

    assert out.shape == (*shape, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.zeros_like(out))
