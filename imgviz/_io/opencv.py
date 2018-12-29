try:
    import cv2
except ImportError:
    cv2 = None


def check_cv2_available():
    if cv2 is None:
        raise ImportError(
            'cv2 is not installed, run following: pip install opencv-python'
        )


def cv_imshow(image, window_name=''):
    check_cv2_available()

    return cv2.imshow(window_name, image[:, :, ::-1])


def cv_waitkey(msec=0):
    check_cv2_available()

    return cv2.waitKey(msec)
