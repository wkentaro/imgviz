import imgviz


def test_arc2017():
    data = imgviz.data.arc2017()
    assert isinstance(data, dict)
