import json
import pathlib
from typing import Any
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

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

    # compose masks to class label image
    class_label: NDArray[np.int32] = np.full(data["rgb"].shape[:2], 0, dtype=np.int32)
    for label_id, mask in zip(data["labels"], data["masks"]):
        class_label[mask == 1] = label_id

    names_file: pathlib.Path = _here / "class_names.txt"
    with open(names_file) as f:
        class_names = [name.strip() for name in f]

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
