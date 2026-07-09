import numpy as np

import imgviz


def test_text() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.text(img, yx=(0, 0), text="TEST", color=(0, 0, 0), size=30)
    assert res.shape == img.shape
    assert res.dtype == img.dtype
    assert not np.array_equal(res, img)


def test_text_size() -> None:
    height, width = imgviz.draw.text_size("Hello", size=20)
    assert isinstance(height, int)
    assert isinstance(width, int)
    assert height > 0
    assert width > 0


def test_text_size_blank_line_adds_height() -> None:
    height_without, _ = imgviz.draw.text_size("a\nb", size=20)
    height_with, _ = imgviz.draw.text_size("a\n\nb", size=20)
    assert height_with > height_without
