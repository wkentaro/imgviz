import numpy as np
import pytest
from numpy.typing import NDArray

from imgviz.draw._ink import Ink
from imgviz.draw._ink import get_pil_ink
from imgviz.draw._ink import require_fill_or_outline


@pytest.mark.parametrize("ink", [None, 5, (255, 0, 0)], ids=["none", "int", "tuple"])
def test_get_pil_ink_passes_through_non_ndarray(ink: Ink | None) -> None:
    assert get_pil_ink(ink) == ink


@pytest.mark.parametrize("size", [3, 4], ids=["rgb", "rgba"])
def test_get_pil_ink_converts_ndarray_to_int_tuple(size: int) -> None:
    ink = np.arange(1, size + 1, dtype=np.uint8)

    result = get_pil_ink(ink)

    assert isinstance(result, tuple)
    assert result == tuple(range(1, size + 1))
    assert all(type(channel) is int for channel in result)  # PIL needs Python ints


@pytest.mark.parametrize(
    "ink",
    [np.array([[1, 2, 3]], dtype=np.uint8), np.array([1, 2], dtype=np.uint8)],
    ids=["2d", "wrong-size"],
)
def test_get_pil_ink_rejects_invalid_ndarray(ink: NDArray[np.uint8]) -> None:
    with pytest.raises(ValueError, match="color ndarray must be 1D with size 3 or 4"):
        get_pil_ink(ink)


def test_require_fill_or_outline_rejects_both_none() -> None:
    with pytest.raises(ValueError, match="at least one of `fill` or `outline`"):
        require_fill_or_outline(None, None)


@pytest.mark.parametrize(
    ("fill", "outline"),
    [((255, 0, 0), None), (None, (0, 0, 0)), ((255, 0, 0), (0, 0, 0))],
    ids=["fill-only", "outline-only", "both"],
)
def test_require_fill_or_outline_accepts_when_set(
    fill: Ink | None, outline: Ink | None
) -> None:
    require_fill_or_outline(fill, outline)
