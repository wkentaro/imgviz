from __future__ import annotations

import pathlib
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from ...io import imread

_here: pathlib.Path = pathlib.Path(__file__).parent


# Code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy  # NOQA
def read_flow(filename: str | pathlib.Path) -> NDArray[np.float32]:
    """Read .flo file in Middlebury format"""

    with open(filename, "rb") as f:
        magic = np.fromfile(f, np.float32, count=1)
        if magic != 202021.25:
            raise OSError(f"invalid .flo file: {filename}")

        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        flow_uv = np.fromfile(f, np.float32, count=2 * w * h)
        # Reshape data into 3D array (columns, rows, bands)
        flow_uv = np.resize(flow_uv, (h, w, 2))
        return flow_uv


class _MiddleburyData(TypedDict):
    rgb: NDArray[np.uint8]
    flow: NDArray[np.float32]


def middlebury() -> _MiddleburyData:
    # http://vision.middlebury.edu/flow/data/

    rgb_file = _here / "grove3.png"
    rgb = imread(rgb_file)

    flow_file = _here / "grove3.flo"
    flow = read_flow(flow_file)

    data: _MiddleburyData = _MiddleburyData(rgb=rgb, flow=flow)
    return data
