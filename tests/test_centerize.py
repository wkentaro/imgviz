import numpy as np

import imgviz


def test_centerize():
    img = np.random.uniform(0, 255, size=(15, 25, 3)).round().astype(np.uint8)

    dst = imgviz.centerize(img, shape=(25, 25), cval=0)
    assert dst.shape == (25, 25, 3)
    assert dst.dtype == img.dtype
    assert (dst[:5] == 0).all()
    assert (dst[-5:] == 0).all()
    np.testing.assert_allclose(dst[5:-5], img)

    dst = imgviz.centerize(img, shape=(15, 35), cval=0)
    assert dst.shape == (15, 35, 3)
    assert dst.dtype == img.dtype
    assert (dst[:, :5] == 0).all()
    assert (dst[:, -5:] == 0).all()
    np.testing.assert_allclose(dst[:, 5:-5], img)
