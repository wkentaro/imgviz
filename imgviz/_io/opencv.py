try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore
import numpy as np  # NOQA


def check_cv2_available():
    if cv2 is None:
        raise ImportError(
            "cv2 is not installed, run following: pip install opencv-python"
        )


def cv_imshow(image, window_name=""):
    # type: (np.ndarray, str) -> None
    """Show image with OpenCV.

    Parameters
    ----------
    image: numpy.ndarray
        Image.

    Returns
    -------
    None

    """
    check_cv2_available()

    return cv2.imshow(window_name, image[:, :, ::-1])


def cv_waitkey(msec=0):
    # type: (int) -> int
    """Wait key for the OpenCV window.

    Parameters
    ----------
    msec: float
        Miliseconds to wait.

    Return
    -------
    keycode: int
        Key code (e.g., ord('q')).

    """
    check_cv2_available()

    return cv2.waitKey(msec)
