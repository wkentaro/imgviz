import numpy as np
import pytest

from imgviz import _utils


def test_apply_mask_none_returns_transformed() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    transformed = np.ones((40, 50, 3), dtype=np.uint8)

    dst = _utils.apply_mask(image=image, transformed=transformed, mask=None)

    assert dst is transformed


def test_apply_mask_composites_within_mask() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    transformed = np.ones((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=bool)
    mask[10:30, 15:35] = True

    dst = _utils.apply_mask(image=image, transformed=transformed, mask=mask)

    np.testing.assert_array_equal(dst[mask], transformed[mask])
    np.testing.assert_array_equal(dst[~mask], image[~mask])


def test_apply_mask_rejects_non_bool_mask() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((40, 50), dtype=np.uint8)

    with pytest.raises(ValueError, match="mask.dtype must be bool"):
        _utils.apply_mask(image=image, transformed=image, mask=mask)


def test_apply_mask_rejects_shape_mismatch() -> None:
    image = np.zeros((40, 50, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=bool)

    with pytest.raises(ValueError, match="mask.shape must be"):
        _utils.apply_mask(image=image, transformed=image, mask=mask)


@pytest.mark.parametrize(
    ("loc", "expected"),
    [
        ("lt", (5, 5)),
        ("rt", (5, 100 - 5 - 30)),
        ("lb", (100 - 5 - 20, 5)),
        ("rb", (100 - 5 - 20, 100 - 5 - 30)),
    ],
)
def test_compute_corner_origin_places_in_each_corner(
    loc: str, expected: tuple[int, int]
) -> None:
    origin = _utils.compute_corner_origin(
        container_size=(100, 100),
        block_size=(20, 30),
        loc=loc,  # type: ignore[arg-type]
        margin=5,
    )

    assert origin == expected


def test_compute_corner_origin_rejects_unsupported_loc() -> None:
    with pytest.raises(ValueError, match="unsupported loc"):
        _utils.compute_corner_origin(
            container_size=(100, 100),
            block_size=(20, 30),
            loc="middle",  # type: ignore[arg-type]
            margin=5,
        )
