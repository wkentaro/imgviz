import matplotlib.pyplot as plt

import imgviz


# sample data of rgb, depth, class label and instance masks
data = imgviz.data.arc2017()
# colorize depth image with JET colormap
depth_viz = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)
# tile visualization
tiled = imgviz.tile([data['rgb'], depth_viz], border=(255, 255, 255))

plt.imshow(tiled)
plt.axis('off')
plt.show()
