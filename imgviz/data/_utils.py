from __future__ import annotations

import pathlib

import numpy as np
from numpy.typing import NDArray


def compose_class_label(
    *,
    shape: tuple[int, int],
    labels: NDArray[np.int32],
    masks: NDArray,
) -> NDArray[np.int32]:
    class_label: NDArray[np.int32] = np.full(shape, 0, dtype=np.int32)
    for label_id, mask in zip(labels, masks):
        class_label[mask == 1] = label_id
    return class_label


def read_class_names(*, path: pathlib.Path) -> list[str]:
    with open(path) as f:
        return [name.strip() for name in f]
