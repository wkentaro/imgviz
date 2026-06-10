import pathlib

import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz


@pytest.mark.parametrize(
    "shape", [(15, 20, 3), (15, 20, 4), (15, 20)], ids=["rgb", "rgba", "grayscale"]
)
def test_imsave_imread_roundtrip(
    tmp_path: pathlib.Path, shape: tuple[int, ...]
) -> None:
    image = np.random.RandomState(0).uniform(0, 255, shape).astype(np.uint8)

    imgviz.io.imsave(tmp_path / "image.png", image)
    read = imgviz.io.imread(tmp_path / "image.png")

    assert read.dtype == np.uint8
    np.testing.assert_array_equal(read, image)


def test_imsave_creates_parent_dirs(tmp_path: pathlib.Path) -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)

    imgviz.io.imsave(tmp_path / "sub" / "deep" / "rgb.png", rgb)

    assert (tmp_path / "sub" / "deep" / "rgb.png").exists()


def test_lblsave_rejects_non_png_extension(tmp_path: pathlib.Path) -> None:
    lbl = np.zeros((4, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match=r"filename must end with '\.png'"):
        imgviz.io.lblsave(tmp_path / "label.jpg", lbl)


def test_lblsave_rejects_non_uint8(tmp_path: pathlib.Path) -> None:
    lbl = np.zeros((4, 4), dtype=np.int32)

    with pytest.raises(ValueError, match=r"lbl\.dtype must be np\.uint8"):
        imgviz.io.lblsave(tmp_path / "label.png", lbl)


def test_lblsave(tmp_path: pathlib.Path) -> None:
    data = imgviz.data.arc2017()

    label_cls: NDArray[np.int32] = data["class_label"]

    assert label_cls.min() == 0
    assert label_cls.max() == 25

    label_cls = label_cls.astype(np.uint8)

    png_file: pathlib.Path = tmp_path / "label_cls.png"
    imgviz.io.lblsave(png_file, label_cls)
    label_cls_read = imgviz.io.imread(png_file)

    np.testing.assert_array_equal(label_cls, label_cls_read)
