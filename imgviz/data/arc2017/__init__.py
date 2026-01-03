import json
import pathlib

import numpy as np

_here: pathlib.Path = pathlib.Path(__file__).parent


def arc2017() -> dict:
    data_file: pathlib.Path = _here / "data.npz"
    data: dict = dict(np.load(data_file))

    # compose masks to class label image
    class_label = np.full(data["rgb"].shape[:2], 0, dtype=np.int32)
    for label_id, mask in zip(data["labels"], data["masks"]):
        class_label[mask == 1] = label_id
    data["class_label"] = class_label

    names_file: pathlib.Path = _here / "class_names.txt"
    with open(names_file) as f:
        class_names = [name.strip() for name in f]
    data["class_names"] = class_names

    data["res4"] = np.load(_here / "res4.npz")["res4"]

    with open(_here / "camera_info.json") as f:
        data["camera_info"] = json.load(f)

    return data
