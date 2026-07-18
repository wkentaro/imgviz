import pathlib

import numpy as np

from imgviz.data import _utils


def test_compose_class_label_paints_last_mask_over_overlap() -> None:
    masks = np.array(
        [
            [[1, 1, 0], [2, 0, 0]],
            [[0, 1, 1], [0, 0, 0]],
        ],
        dtype=np.int32,
    )
    labels = np.array([7, 3], dtype=np.int32)

    class_label = _utils.compose_class_label(shape=(2, 3), labels=labels, masks=masks)

    np.testing.assert_array_equal(class_label, [[7, 3, 3], [0, 0, 0]])
    assert class_label.dtype == np.int32


def test_read_class_names_strips_whitespace(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "class_names.txt"
    path.write_text("  cat \ndog\n")

    class_names = _utils.read_class_names(path)

    assert class_names == ["cat", "dog"]
