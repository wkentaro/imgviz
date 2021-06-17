#!/usr/bin/env python
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
depth = data["depth"]
depthviz = imgviz.depth2rgb(depth, min_value=0.3, max_value=1)

# colorize label image
class_label = data["class_label"]
labelviz = imgviz.label2rgb(class_label, label_names=data["class_names"], font_size=20)

# instance bboxes
rgb = data["rgb"]
bboxes = data["bboxes"].astype(int)
labels = data["labels"]
captions = [data["class_names"][l] for l in labels]
bboxviz = imgviz.instances2rgb(image=rgb, bboxes=bboxes, labels=labels, captions=captions)

# instance masks
masks = data["masks"] == 1
maskviz = imgviz.instances2rgb(image=rgb, masks=masks, labels=labels, captions=captions)

# tile instance masks
insviz = [(rgb * m[:, :, None])[b[0] : b[2], b[1] : b[3]] for b, m in zip(bboxes, masks)]
insviz = imgviz.tile(imgs=insviz, border=(255, 255, 255))

# tile visualization
tiled = imgviz.tile(
    [rgb, depthviz, labelviz, bboxviz, maskviz, insviz],
    shape=(1, 6),
    border=(255, 255, 255),
    border_width=5,
)
# }} GETTING_STARTED
# -----------------------------------------------------------------------------

out_file = osp.join(here, ".readme/getting_started.jpg")
imgviz.io.imsave(out_file, tiled)

img = imgviz.io.imread(out_file)
plt.imshow(img)
plt.axis("off")
plt.show()
