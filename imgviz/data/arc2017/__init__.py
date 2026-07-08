import json
import pathlib
from typing import Any
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .. import _utils

_here: pathlib.Path = pathlib.Path(__file__).parent


class _Arc2017Data(TypedDict):
    rgb: NDArray[np.uint8]
    depth: NDArray[np.float32]
    bboxes: NDArray[np.float32]
    labels: NDArray[np.int32]
    masks: NDArray[np.int32]
    class_label: NDArray[np.int32]
    class_names: list[str]
    res4: NDArray[np.float32]
    camera_info: dict[str, Any]


def arc2017() -> _Arc2017Data:
    data_file: pathlib.Path = _here / "data.npz"
    data: dict = dict(np.load(data_file))

    class_label = _utils.compose_class_label(
        shape=data["rgb"].shape[:2], labels=data["labels"], masks=data["masks"]
    )
    class_names = _utils.read_class_names(_here / "class_names.txt")

    res4: NDArray[np.float32] = np.load(_here / "res4.npz")["res4"]

    with open(_here / "camera_info.json") as f:
        camera_info: dict[str, Any] = json.load(f)

    return _Arc2017Data(
        rgb=data["rgb"],
        depth=data["depth"],
        bboxes=data["bboxes"],
        labels=data["labels"],
        masks=data["masks"],
        class_label=class_label,
        class_names=class_names,
        res4=res4,
        camera_info=camera_info,
    )
