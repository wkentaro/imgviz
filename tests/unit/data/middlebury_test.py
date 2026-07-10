import pathlib

import numpy as np
import pytest

from imgviz.data.middlebury import read_flow


def test_read_flow_rejects_invalid_magic(tmp_path: pathlib.Path) -> None:
    flow_file = tmp_path / "invalid.flo"
    np.array([1.0], dtype=np.float32).tofile(flow_file)

    with pytest.raises(OSError, match="invalid .flo file"):
        read_flow(flow_file)
