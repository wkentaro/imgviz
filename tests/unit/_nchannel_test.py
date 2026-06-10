import numpy as np
import pytest
import sklearn.decomposition
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def nchannel() -> NDArray[np.float32]:
    rng = np.random.RandomState(0)
    return rng.rand(8, 8, 16).astype(np.float32)


def test_nchannel2rgb_returns_uint8_rgb(nchannel: NDArray[np.float32]) -> None:
    out = imgviz.nchannel2rgb(nchannel)

    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8
    assert out.max() > out.min()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_nchannel2rgb_float_dtype(
    nchannel: NDArray[np.float32], dtype: type[np.floating]
) -> None:
    out = imgviz.nchannel2rgb(nchannel, dtype=dtype)

    assert out.shape == (8, 8, 3)
    assert out.dtype == dtype


def test_nchannel2rgb_rejects_invalid_output_dtype(
    nchannel: NDArray[np.float32],
) -> None:
    with pytest.raises(ValueError, match="dtype must be floating"):
        imgviz.nchannel2rgb(nchannel, dtype=np.int32)  # type: ignore[call-overload]


def test_nchannel2rgb_is_deterministic(nchannel: NDArray[np.float32]) -> None:
    out1 = imgviz.nchannel2rgb(nchannel)
    out2 = imgviz.nchannel2rgb(nchannel)

    np.testing.assert_array_equal(out1, out2)


def test_nchannel2rgb_with_prefit_pca(nchannel: NDArray[np.float32]) -> None:
    pca = sklearn.decomposition.PCA(n_components=3, random_state=0)
    pca.fit(nchannel.reshape(-1, nchannel.shape[-1]))

    out = imgviz.nchannel2rgb(nchannel, pca=pca)

    assert out.shape == (8, 8, 3)
    assert out.dtype == np.uint8


def test_Nchannel2Rgb_exposes_pca() -> None:
    pca = sklearn.decomposition.PCA(n_components=3, random_state=0)

    converter = imgviz.Nchannel2Rgb(pca=pca)

    assert converter.pca is pca


def test_Nchannel2Rgb_reuses_cached_range_across_frames(
    nchannel: NDArray[np.float32],
) -> None:
    pca = sklearn.decomposition.PCA(n_components=3, random_state=0)
    pca.fit(nchannel.reshape(-1, nchannel.shape[-1]))

    converter = imgviz.Nchannel2Rgb(pca=pca)
    converter(nchannel)

    second_frame = nchannel * 2.0
    out_reused = converter(second_frame)
    out_fresh = imgviz.nchannel2rgb(second_frame, pca=pca)

    assert out_reused.shape == (8, 8, 3)
    assert out_reused.dtype == np.uint8
    assert not np.array_equal(out_reused, out_fresh)


def test_nchannel2rgb_rejects_non_3d() -> None:
    with pytest.raises(ValueError, match="nchannel.ndim must be 3"):
        imgviz.nchannel2rgb(np.zeros((4, 4), dtype=np.float32))


def test_nchannel2rgb_rejects_non_floating() -> None:
    with pytest.raises(ValueError, match="nchannel.dtype must be floating"):
        imgviz.nchannel2rgb(np.zeros((4, 4, 3), dtype=np.uint8))


# res4's float32 values overflow in sklearn's PCA matmul; the warning is benign.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_nchannel2rgb_matches_example() -> None:
    res4 = imgviz.data.arc2017()["res4"]

    out = imgviz.nchannel2rgb(res4)

    assert out.shape == (*res4.shape[:2], 3)
    assert out.dtype == np.uint8
