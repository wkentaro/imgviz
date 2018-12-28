import os.path as osp

import matplotlib.pyplot as plt

here = osp.dirname(osp.abspath(__file__))  # NOQA

# -----------------------------------------------------------------------------
# GETTING_STARTED {{
import imgviz


# sample data of rgb, depth, class label and instance masks
data = imgviz.data.arc2017()
# colorize depth image with JET colormap
depth_viz = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)
# tile visualization
tiled = imgviz.tile([data['rgb'], depth_viz], border=(255, 255, 255))
# }} GETTING_STARTED
# -----------------------------------------------------------------------------

out_file = osp.join(here, '.readme/getting_started.jpg')
plt.imsave(out_file, tiled)
plt.imshow(plt.imread(out_file))
plt.axis('off')
plt.show()
