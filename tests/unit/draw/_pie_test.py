import math

import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def white_image() -> NDArray[np.uint8]:
    return np.full((200, 200, 3), 255, dtype=np.uint8)


def test_pie_preserves_shape_and_dtype(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.pie(
        white_image,
        center=(100, 100),
        diameter=80,
        fills=[(255, 0, 0)],
    )
    assert res.shape == white_image.shape
    assert res.dtype == white_image.dtype
    assert not np.array_equal(res, white_image)


def test_pie_empty_fills_is_noop(white_image: NDArray[np.uint8]) -> None:
    res = imgviz.draw.pie(
        white_image,
        center=(100, 100),
        diameter=80,
        fills=[],
    )
    assert np.array_equal(res, white_image)


def test_pie_single_fill_colors_disc(white_image: NDArray[np.uint8]) -> None:
    center = (100, 100)
    diameter = 80
    fill = (0, 0, 255)
    res = imgviz.draw.pie(
        white_image,
        center=center,
        diameter=diameter,
        fills=[fill],
    )
    # The center pixel must be the fill color.
    cy, cx = center
    assert tuple(res[cy, cx]) == fill


def test_pie_three_wedges_distinct_colors(white_image: NDArray[np.uint8]) -> None:
    # Three equal wedges (120 deg each) clockwise from 12 o'clock:
    #   Wedge 0 (red):   0-120  deg from 12 -- midpoint at  60 deg (upper-right)
    #   Wedge 1 (green): 120-240 deg from 12 -- midpoint at 180 deg (bottom)
    #   Wedge 2 (blue):  240-360 deg from 12 -- midpoint at 300 deg (upper-left)
    fills: list[tuple[int, int, int]] = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
    ]
    cy, cx = 100, 100
    res = imgviz.draw.pie(
        white_image,
        center=(cy, cx),
        diameter=80,
        fills=fills,
    )

    r = 25  # sample radius, well inside the disc

    def sample_at(angle_from_12_deg: float) -> tuple[int, ...]:
        # Shift from clockwise-from-12 to standard math angle (0 = east), in radians.
        angle_rad = math.radians(angle_from_12_deg - 90)
        sy = int(cy + r * math.sin(angle_rad))
        sx = int(cx + r * math.cos(angle_rad))
        return tuple(res[sy, sx])

    assert sample_at(60) == fills[0], "midpoint of wedge 0 should be red"
    assert sample_at(180) == fills[1], "midpoint of wedge 1 should be green"
    assert sample_at(300) == fills[2], "midpoint of wedge 2 should be blue"


def test_pie_outline_is_drawn(white_image: NDArray[np.uint8]) -> None:
    outline = (0, 0, 0)
    res = imgviz.draw.pie(
        white_image,
        center=(100, 100),
        diameter=80,
        fills=[(200, 200, 200)],
        outline=outline,
        width=2,
    )
    # The outline color must appear somewhere in the output.
    outline_arr = np.array(outline, dtype=np.uint8)
    mask = (res == outline_arr).all(axis=-1)
    assert mask.any(), "outline pixels not found in output"
