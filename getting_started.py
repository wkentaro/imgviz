import matplotlib.pyplot as plt

import imgviz


data = imgviz.data.arc2017()

depth_viz = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)

tiled = imgviz.tile([data['rgb'], depth_viz], border=(255, 255, 255))

plt.imshow(tiled)
plt.axis('off')
plt.show()
