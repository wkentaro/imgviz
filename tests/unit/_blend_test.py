import numpy as np
import pytest

from imgviz._blend import blend


def test_blend_rounds_to_nearest() -> None:
    under = np.full((2, 2, 3), 101, dtype=np.uint8)

    out = blend(under, (202, 202, 202), alpha=0.5)

    # 0.5 * 101 + 0.5 * 202 = 151.5 must round to 152, not truncate to 151.
    assert out.dtype == np.uint8
    assert (out == 152).all()


def test_blend_accepts_per_pixel_alpha_and_image_over() -> None:
    under = np.zeros((1, 2, 3), dtype=np.uint8)
    over = np.full((1, 2, 3), 200, dtype=np.uint8)
    alpha = np.array([[[0.0], [1.0]]])

    out = blend(under, over, alpha)

    assert out[0, 0].tolist() == [0, 0, 0]
    assert out[0, 1].tolist() == [200, 200, 200]


@pytest.mark.parametrize("alpha", [-0.1, 1.1])
def test_blend_rejects_out_of_range_alpha(alpha: float) -> None:
    under = np.zeros((1, 1, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="alpha must be in range"):
        blend(under, (0, 0, 0), alpha)
