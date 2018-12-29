# flake8: noqa

import os.path as osp

import matplotlib.pyplot as plt

here = osp.dirname(osp.abspath(__file__))  # NOQA

# -----------------------------------------------------------------------------
# GETTING_STARTED {{
import imgviz


# sample data of rgb, depth, class label and instance masks
data = imgviz.data.arc2017()

# colorize depth image with JET colormap
depthviz = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)

# colorize label image
labelviz = imgviz.label2rgb(data['class_label'], label_names=data['class_names'])

# tile instance masks
bboxes = data['bboxes'].astype(int)
insviz = [data['rgb'][b[0]:b[2], b[1]:b[3]] for b in bboxes]
insviz = imgviz.tile(imgs=insviz, border=(255, 255, 255))

# tile visualization
tiled = imgviz.tile(
    [data['rgb'], depthviz, labelviz, insviz],
    shape=(1, 4),
    border=(255, 255, 255),
)
# }} GETTING_STARTED
# -----------------------------------------------------------------------------

out_file = osp.join(here, '.readme/getting_started.jpg')
img = imgviz.io.imread(out_file)
img = imgviz.resize(img, width=1280)
imgviz.io.pyglet_imshow(img)
imgviz.io.pyglet_run()
