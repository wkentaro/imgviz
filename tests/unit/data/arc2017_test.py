import imgviz


def test_arc2017() -> None:
    data = imgviz.data.arc2017()
    assert isinstance(data, dict)
