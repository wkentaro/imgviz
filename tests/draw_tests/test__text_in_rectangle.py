import numpy as np

import imgviz


def test_text_in_rectangle() -> None:
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    res = imgviz.draw.text_in_rectangle(
        img,
        loc="lt",
        text="Hello",
        size=10,
        background=(0, 0, 0),
    )
    assert res.shape == img.shape
    assert res.dtype == np.uint8
    assert not np.array_equal(res, img)
