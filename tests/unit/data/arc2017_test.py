import numpy as np

import imgviz


def test_arc2017() -> None:
    data = imgviz.data.arc2017()
    assert isinstance(data, dict)


def test_arc2017_wires_expected_fields() -> None:
    data = imgviz.data.arc2017()

    height, width = 480, 640
    num_instances = 8

    assert data["rgb"].dtype == np.uint8
    assert data["rgb"].shape == (height, width, 3)

    assert data["depth"].dtype == np.float32
    assert data["depth"].shape == (height, width)

    assert data["bboxes"].dtype == np.float32
    assert data["bboxes"].shape == (num_instances, 4)

    assert data["labels"].dtype == np.int32
    assert data["labels"].shape == (num_instances,)

    assert data["masks"].dtype == np.int32
    assert data["masks"].shape == (num_instances, height, width)

    assert data["class_label"].dtype == np.int32
    assert data["class_label"].shape == (height, width)

    assert isinstance(data["class_names"], list)
    assert len(data["class_names"]) == 41
    assert all(isinstance(name, str) for name in data["class_names"])

    assert data["res4"].dtype == np.float32
    assert data["res4"].shape == (30, 40, 1024)

    assert isinstance(data["camera_info"], dict)
