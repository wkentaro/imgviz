import imgviz
import numpy as np
import matplotlib.pyplot as plt

img = np.zeros((600, 400, 3), np.uint8)

x1, y1 = (100, 100)
x2, y2 = (300, 200)

img = imgviz.draw.text_in_rectangle(img, loc="lt", text="long texttttttttttttttttttttttt", size=30, background=(0, 0, 255), aabb1=(y1, x1), aabb2=(y2, x2))
img = imgviz.draw.text_in_rectangle(img, loc="lt", text="\nshort", size=30, background=(255, 0, 0), aabb1=(y1, x1), aabb2=(y2, x2))
img = imgviz.draw.rectangle(img, (y1, x1), (y2, x2), outline=(0, 255, 0), width=1)

plt.figure()
plt.imshow(img)
plt.show()
