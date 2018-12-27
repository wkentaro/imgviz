try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import PIL.Image


def resize(
    src,
    height=None,
    width=None,
    interpolation='linear',
    backend='auto',
):
    if not isinstance(src, np.ndarray):
        raise TypeError('src type must be numpy.ndarray')

    if backend == 'auto':
        backend = 'pillow' if cv2 is None else 'opencv'

    src_height, src_width = src.shape[:2]
    if isinstance(width, float):
        scale_width = width
        width = int(round(scale_width * src_width))
    if isinstance(height, float):
        scale_height = height
        height = int(round(scale_height * src_height))
    if height is None:
        assert width is not None
        scale_height = 1. * width / src_width
        height = int(round(scale_height * src_height))
    if width is None:
        assert height is not None
        scale_width = 1. * height / src_height
        width = int(round(scale_width * src_width))

    if backend == 'pillow':
        if interpolation == 'linear':
            interpolation = PIL.Image.LINEAR
        elif interpolation == 'nearest':
            interpolation = PIL.Image.NEAREST
        else:
            raise ValueError(
                'unsupported interpolation: {}'.format(interpolation)
            )

        src = PIL.Image.fromarray(src)
        dst = src.resize((width, height), interpolation)
        dst = np.asarray(dst)
    elif backend == 'opencv':
        if interpolation == 'linear':
            interpolation = cv2.INTER_LINEAR
        elif interpolation == 'nearest':
            interpolation = cv2.INTER_NEAREST
        else:
            raise ValueError(
                'unsupported interpolation: {}'.format(interpolation)
            )

        dst = cv2.resize(src, (width, height), interpolation=interpolation)
    else:
        raise ValueError('unsupported backend: {}'.format(backend))

    return dst
