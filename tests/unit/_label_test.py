from typing import Literal
from typing import TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.fixture
def labeled_square() -> NDArray[np.int32]:
    label = np.zeros((20, 20), dtype=np.int32)
    label[5:15, 5:15] = 1
    return label


def test_label2rgb() -> None:
    data = imgviz.data.arc2017()
    H, W = data["class_label"].shape[:2]

    labelviz = imgviz.label2rgb(label=data["class_label"])
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)

    labelviz = imgviz.label2rgb(label=data["class_label"], image=data["rgb"])
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)

    labelviz = imgviz.label2rgb(
        label=data["class_label"],
        image=data["rgb"],
        label_names=data["class_names"],
    )
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)

    labelviz = imgviz.label2rgb(
        label=data["class_label"],
        image=data["rgb"],
        label_names=data["class_names"],
        alpha=[0.5 for _ in data["class_names"]],
    )
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)

    labelviz = imgviz.label2rgb(
        label=data["class_label"],
        image=data["rgb"],
        label_names=data["class_names"],
        alpha={i: 0.5 for i in range(len(data["class_names"]))},
    )
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)

    # Test all legend locations
    Loc: TypeAlias = Literal["lt", "rt", "lb", "rb", "centroid"]
    for loc in Loc.__args__:
        labelviz = imgviz.label2rgb(
            label=data["class_label"],
            image=data["rgb"],
            label_names=data["class_names"],
            loc=loc,
        )
        assert labelviz.dtype == np.uint8
        assert labelviz.shape == (H, W, 3)


def test_label2rgb_all_unlabeled() -> None:
    label = np.full((8, 8), -1, dtype=np.int32)
    labelviz = imgviz.label2rgb(label=label)
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (8, 8, 3)

    image = np.zeros((8, 8, 3), dtype=np.uint8)
    labelviz = imgviz.label2rgb(label=label, image=image)
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (8, 8, 3)


def test_label_colormap_is_cached_and_readonly() -> None:
    cmap = imgviz.label_colormap()
    assert cmap.shape == (256, 3)
    assert cmap.dtype == np.uint8
    assert imgviz.label_colormap() is cmap
    with pytest.raises(ValueError, match="read-only"):
        cmap[0] = (1, 2, 3)


def test_label2rgb_casts_bool_label() -> None:
    label = np.array([[True, False], [False, True]], dtype=bool)

    labelviz = imgviz.label2rgb(label=label)

    cmap = imgviz.label_colormap()
    np.testing.assert_array_equal(labelviz[0, 0], cmap[1])
    np.testing.assert_array_equal(labelviz[0, 1], cmap[0])


def test_label2rgb_accepts_gray_image() -> None:
    label = np.array([[0, 1], [1, 0]], dtype=np.int32)
    gray = np.full((2, 2), 100, dtype=np.uint8)

    labelviz = imgviz.label2rgb(label=label, image=gray)

    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (2, 2, 3)
    # gray 100 broadcast to 3 channels, then 0.5-blended with the label color:
    # label 0 = black -> 50; label 1 = (128, 0, 0) -> (114, 50, 50).
    np.testing.assert_array_equal(labelviz[0, 0], [50, 50, 50])
    np.testing.assert_array_equal(labelviz[0, 1], [114, 50, 50])


def test_label2rgb_rejects_out_of_range_alpha() -> None:
    label = np.array([[0, 1]], dtype=np.int32)
    with pytest.raises(ValueError, match=r"alpha values must be in \[0, 1\]"):
        imgviz.label2rgb(label=label, alpha=[0.5, 1.5])


def test_label2rgb_dict_label_names_centroid(labeled_square: NDArray[np.int32]) -> None:
    labelviz = imgviz.label2rgb(
        label=labeled_square,
        label_names={0: "bg", 1: "obj"},
        loc="centroid",
    )

    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (20, 20, 3)
    # Text was drawn from the dict names, so it differs from the plain render.
    assert not np.array_equal(labelviz, imgviz.label2rgb(label=labeled_square))


def test_label2rgb_thresh_suppress_draws_no_text(
    labeled_square: NDArray[np.int32],
) -> None:
    suppressed = imgviz.label2rgb(
        label=labeled_square,
        label_names={0: "bg", 1: "obj"},
        loc="centroid",
        thresh_suppress=1.0,
    )

    np.testing.assert_array_equal(suppressed, imgviz.label2rgb(label=labeled_square))
