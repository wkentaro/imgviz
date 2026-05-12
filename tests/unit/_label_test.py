from typing import Literal
from typing import TypeAlias

import numpy as np

import imgviz


def test_label2rgb():
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
