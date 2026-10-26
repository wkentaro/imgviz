import numpy as np
import pytest

import imgviz


def test_blend_public_call_styles() -> None:
    under = np.full((2, 3, 3), 10, dtype=np.uint8)
    over = (111, 210, 11)
    expected = np.full_like(under, (60, 110, 10))
    assert "blend" in imgviz.__all__
    for result in (
        imgviz.blend(under, over, 0.5),
        imgviz.blend(under, over, alpha=0.5),
        imgviz.blend(under=under, over=over, alpha=0.5),
    ):
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.uint8
        assert not np.shares_memory(result, under)
    np.testing.assert_array_equal(under, np.full_like(under, 10))


def test_blend_array_alpha() -> None:
    under = np.zeros((2, 1, 3), dtype=np.uint8)
    alpha = np.array([0.0, 1.0]).reshape(2, 1, 1)
    result = imgviz.blend(under, (10, 20, 30), alpha)
    np.testing.assert_array_equal(result, [[[0, 0, 0]], [[10, 20, 30]]])


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan")])
def test_blend_invalid_alpha(*, alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha must be in range"):
        imgviz.blend(np.zeros((1, 1, 3), dtype=np.uint8), (255, 255, 255), alpha)
