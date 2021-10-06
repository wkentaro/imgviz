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
        alpha=0.5
    )
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)
    labelviz = imgviz.label2rgb(
        label=data["class_label"],
        image=data["rgb"],
        label_names=data["class_names"],
        alpha=[0.5 for i in range(np.max(data["class_label"]) + 1)]
    )
    assert labelviz.dtype == np.uint8
    assert labelviz.shape == (H, W, 3)
