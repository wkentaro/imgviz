import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def black_image() -> NDArray[np.uint8]:
    return np.zeros((200, 200, 3), dtype=np.uint8)


def make_flag_color(flag_index: int) -> NDArray[np.uint8]:
    return imgviz.label_colormap()[1 + flag_index]


def test_flags2rgb_preserves_shape_and_dtype(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[True, False]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken", "occluded"],
    )
    assert res.shape == black_image.shape
    assert res.dtype == black_image.dtype


def test_flags2rgb_promotes_grayscale(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image[:, :, 0],
        flags=np.array([[True]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken"],
    )
    assert res.shape == black_image.shape


def test_flags2rgb_on_single_flag_draws_solid_disc(
    black_image: NDArray[np.uint8],
) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[False, True]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken", "occluded"],
    )
    assert np.array_equal(res[100, 95], make_flag_color(1))
    assert np.array_equal(res[100, 105], make_flag_color(1))


def test_flags2rgb_on_multiple_flags_split_into_wedges(
    black_image: NDArray[np.uint8],
) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[True, True]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken", "occluded"],
    )
    assert np.array_equal(res[100, 105], make_flag_color(0))
    assert np.array_equal(res[100, 95], make_flag_color(1))


def test_flags2rgb_on_zero_flags_draws_no_pie(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[False, False]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken", "occluded"],
    )
    assert np.array_equal(res[60:140, 60:140], black_image[60:140, 60:140])


def test_flags2rgb_all_draws_off_wedges_gray(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[True, False]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken", "occluded"],
        wedges="all",
    )
    assert np.array_equal(res[100, 105], make_flag_color(0))
    assert np.array_equal(res[100, 95], (200, 200, 200))


def test_flags2rgb_legend_lists_all_flags(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[False, False]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken", "occluded"],
        loc="lt",
    )
    legend_corner = res[:100, :100]
    assert (legend_corner == make_flag_color(0)).all(axis=2).any()
    assert (legend_corner == make_flag_color(1)).all(axis=2).any()


def test_flags2rgb_all_legend_gains_off_entry(black_image: NDArray[np.uint8]) -> None:
    flags = np.array([[True, True]])
    centers = np.array([[100.0, 100.0]])
    flag_names = ["broken", "occluded"]

    res_on = imgviz.flags2rgb(
        black_image, flags=flags, centers=centers, flag_names=flag_names, loc="lt"
    )
    res_all = imgviz.flags2rgb(
        black_image,
        flags=flags,
        centers=centers,
        flag_names=flag_names,
        wedges="all",
        loc="lt",
    )
    assert not (res_on[:100, :100] == (200, 200, 200)).all(axis=2).any()
    assert (res_all[:100, :100] == (200, 200, 200)).all(axis=2).any()


def test_flags2rgb_custom_flag_colors(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[True]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken"],
        flag_colors=np.array([[255, 255, 0]], dtype=np.uint8),
    )
    assert np.array_equal(res[100, 100], (255, 255, 0))


def test_flags2rgb_outline_defaults_to_white(black_image: NDArray[np.uint8]) -> None:
    res = imgviz.flags2rgb(
        black_image,
        flags=np.array([[True]]),
        centers=np.array([[100.0, 100.0]]),
        flag_names=["broken"],
    )
    disc = res[85:115, 85:115]
    assert (disc == (255, 255, 255)).all(axis=2).any()


def test_flags2rgb_custom_outline_color(black_image: NDArray[np.uint8]) -> None:
    flags = np.array([[True]])
    centers = np.array([[100.0, 100.0]])
    flag_names = ["broken"]
    white = imgviz.flags2rgb(
        black_image,
        flags=flags,
        centers=centers,
        flag_names=flag_names,
        outline=(255, 255, 255),
        outline_width=3,
    )
    black = imgviz.flags2rgb(
        black_image,
        flags=flags,
        centers=centers,
        flag_names=flag_names,
        outline=(0, 0, 0),
        outline_width=3,
    )

    assert not np.array_equal(white, black)
    disc = slice(85, 115)
    assert (white[disc, disc] == (255, 255, 255)).all(axis=2).any()
    assert not (black[disc, disc] == (255, 255, 255)).all(axis=2).any()


def test_flags2rgb_validates_inputs(black_image: NDArray[np.uint8]) -> None:
    flags = np.array([[True]])
    centers = np.array([[100.0, 100.0]])
    flag_names = ["broken"]

    with pytest.raises(TypeError, match="image must be a numpy array"):
        imgviz.flags2rgb(
            [[0]],  # ty: ignore[invalid-argument-type]
            flags=flags,
            centers=centers,
            flag_names=flag_names,
        )
    with pytest.raises(ValueError, match="image dtype must be np.uint8"):
        imgviz.flags2rgb(
            black_image.astype(float),  # ty: ignore[invalid-argument-type]
            flags=flags,
            centers=centers,
            flag_names=flag_names,
        )
    with pytest.raises(ValueError, match="image must be 2 or 3 dimensional"):
        imgviz.flags2rgb(
            np.zeros((10,), dtype=np.uint8),
            flags=flags,
            centers=centers,
            flag_names=flag_names,
        )
    with pytest.raises(TypeError, match="flags must be a numpy array"):
        imgviz.flags2rgb(
            black_image,
            flags=[[True]],  # ty: ignore[invalid-argument-type]
            centers=centers,
            flag_names=flag_names,
        )
    with pytest.raises(ValueError, match="flags dtype must be bool"):
        imgviz.flags2rgb(
            black_image,
            flags=flags.astype(int),  # ty: ignore[invalid-argument-type]
            centers=centers,
            flag_names=flag_names,
        )
    with pytest.raises(ValueError, match="flags must be 2 dimensional"):
        imgviz.flags2rgb(
            black_image,
            flags=flags[0],
            centers=centers,
            flag_names=flag_names,
        )
    with pytest.raises(ValueError, match="centers shape must be"):
        imgviz.flags2rgb(
            black_image,
            flags=flags,
            centers=np.array([[100.0, 100.0], [50.0, 50.0]]),
            flag_names=flag_names,
        )
    with pytest.raises(ValueError, match="flag_names must have one name per flag"):
        imgviz.flags2rgb(
            black_image,
            flags=flags,
            centers=centers,
            flag_names=["broken", "occluded"],
        )
    with pytest.raises(ValueError, match="flag_colors shape must be"):
        imgviz.flags2rgb(
            black_image,
            flags=flags,
            centers=centers,
            flag_names=flag_names,
            flag_colors=np.array([[255, 255, 0], [0, 255, 255]], dtype=np.uint8),
        )
    with pytest.raises(ValueError, match="unsupported wedges"):
        imgviz.flags2rgb(
            black_image,
            flags=flags,
            centers=centers,
            flag_names=flag_names,
            wedges="off",  # ty: ignore[invalid-argument-type]
        )
