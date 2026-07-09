import pathlib
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

from .. import _utils

_here: pathlib.Path = pathlib.Path(__file__).parent


class _VocData(TypedDict):
    rgb: NDArray[np.uint8]
    bboxes: NDArray[np.floating]
    labels: NDArray[np.int32]
    masks: NDArray[np.bool_]
    class_label: NDArray[np.int32]
    class_names: list[str]


def voc() -> _VocData:
    data_file: pathlib.Path = _here / "data.npz"
    data: dict = dict(np.load(data_file))

    class_label = _utils.compose_class_label(
        shape=data["rgb"].shape[:2], labels=data["labels"], masks=data["masks"]
    )
    class_names = _utils.read_class_names(_here / "class_names.txt")

    return _VocData(
        rgb=data["rgb"],
        bboxes=data["bboxes"],
        labels=data["labels"],
        masks=data["masks"],
        class_label=class_label,
        class_names=class_names,
    )
