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

# instance bboxes
bboxes = data['bboxes'].astype(int)
captions = [data['class_names'][l] for l in data['labels']]
bboxviz = imgviz.instances2rgb(
    image=data['rgb'], bboxes=bboxes, labels=data['labels'], captions=captions
)

# instance masks
masks = data['masks'] == 1
maskviz = imgviz.instances2rgb(
    image=data['rgb'], masks=masks, labels=data['labels'], captions=captions
)

# tile instance masks
insviz = [
    (data['rgb'] * mask[:, :, None])[y1:y2, x1:x2]
    for (y1, x1, y2, x2), mask in zip(bboxes, masks)
]
insviz = imgviz.tile(imgs=insviz, border=(255, 255, 255))

# tile visualization
tiled = imgviz.tile(
    [data['rgb'], depthviz, labelviz, bboxviz, maskviz, insviz],
    shape=(2, 3),
    border=(255, 255, 255),
)
# }} GETTING_STARTED
# -----------------------------------------------------------------------------

out_file = osp.join(here, '.readme/getting_started.jpg')
imgviz.io.imsave(out_file, tiled)

img = imgviz.io.imread(out_file)
plt.imshow(img)
plt.axis('off')
plt.show()
