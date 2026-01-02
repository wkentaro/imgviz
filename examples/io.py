import pathlib

import numpy as np
from numpy.typing import NDArray

import imgviz


def main() -> None:
    assert imgviz.__file__ is not None
    image_file: pathlib.Path = (
        pathlib.Path(imgviz.__file__).parent / "data" / "arc2017" / "rgb.jpg"
    )

    rgb: NDArray[np.uint8] = imgviz.io.imread(str(image_file))

    imgviz.io.imshow(rgb)


if __name__ == "__main__":
    main()
