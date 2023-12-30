import numpy as np

import imgviz


def test_masks_to_bboxes():
    data = imgviz.data.arc2017()

    class_label = data["class_label"]
    masks = [class_label == label_id for label_id in np.unique(class_label)]
    bboxes = imgviz.instances.masks_to_bboxes(masks)

    assert len(bboxes) == len(masks)
    assert bboxes.shape[1] == 4

    ymin = bboxes[:, 0]
    xmin = bboxes[:, 1]
    ymax = bboxes[:, 2]
    xmax = bboxes[:, 3]
    height, width = class_label.shape
    assert ((0 <= ymin) & (ymin <= height - 1)).all()
    assert ((0 <= ymax) & (ymax <= height - 1)).all()
    assert ((0 <= xmin) & (xmin <= width - 1)).all()
    assert ((0 <= xmax) & (xmax <= width - 1)).all()
