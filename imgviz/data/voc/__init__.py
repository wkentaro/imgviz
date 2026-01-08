import pathlib
from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

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

    # compose masks to class label image
    class_label: NDArray[np.int32] = np.full(data["rgb"].shape[:2], 0, dtype=np.int32)
    for label_id, mask in zip(data["labels"], data["masks"]):
        class_label[mask == 1] = label_id

    names_file: pathlib.Path = _here / "class_names.txt"
    with open(names_file) as f:
        class_names = [name.strip() for name in f]

    return _VocData(
        rgb=data["rgb"],
        bboxes=data["bboxes"],
        labels=data["labels"],
        masks=data["masks"],
        class_label=class_label,
        class_names=class_names,
    )
