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
crops = []
for bbox, mask in zip(data['bboxes'], data['masks']):
    y1, x1, y2, x2 = bbox.astype(int)
    rgb_crop = data['rgb'][y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    rgb_crop[mask_crop != 1] = 0
    crops.append(rgb_crop)
insviz = imgviz.tile(imgs=crops, border=(255, 255, 255))

# tile visualization
tiled = imgviz.tile(
    [data['rgb'], depthviz, labelviz, insviz],
    shape=(1, 4),
    border=(255, 255, 255),
)
# }} GETTING_STARTED
# -----------------------------------------------------------------------------

out_file = osp.join(here, '.readme/getting_started.jpg')
plt.imsave(out_file, tiled)
plt.imshow(plt.imread(out_file))
plt.axis('off')
plt.show()
