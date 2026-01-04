import pathlib

import numpy as np

import imgviz


def test_lblsave(tmp_path: pathlib.Path) -> None:
    data: dict = imgviz.data.arc2017()

    label_cls: np.ndarray = data["class_label"]

    assert label_cls.min() == 0
    assert label_cls.max() == 25

    label_cls = label_cls.astype(np.uint8)

    png_file: pathlib.Path = tmp_path / "label_cls.png"
    imgviz.io.lblsave(png_file, label_cls)
    label_cls_read = imgviz.io.imread(png_file)

    np.testing.assert_allclose(label_cls, label_cls_read)
