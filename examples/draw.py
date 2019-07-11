import os.path as osp

import matplotlib.pyplot as plt

import imgviz


here = osp.dirname(osp.abspath(__file__))


if __name__ == '__main__':
    img = imgviz.data.lena()
    H, W = img.shape[:2]
    viz = img

    y1, x1 = 200, 180
    y2, x2 = 400, 380
    viz = imgviz.draw.rectangle(
        viz, (y1, x1), (y2, x2), outline=(255, 255, 255), width=5
    )
    viz = imgviz.instances2rgb(
        viz,
        labels=[1],
        bboxes=[(y1, x1, y2, x2)],
        captions=['face'],
        font_size=30,
        colormap=[(255, 255, 255)],
    )

    # eye, eye, nose, mouse, mouse
    xys = [(265, 265), (330, 265), (315, 320), (270, 350), (320, 350)]
    colors = imgviz.label_colormap(value=255)[1:]
    shapes = ['star', 'star', 'rectangle', 'circle', 'circle']
    for xy, color, shape in zip(xys, colors, shapes):
        size = 20
        color = tuple(color)
        if shape == 'star':
            viz = imgviz.draw.star(
                viz, center=(xy[1], xy[0]), size=1.2 * size, fill=color)
        elif shape == 'circle':
            viz = imgviz.draw.circle(
                viz, center=(xy[1], xy[0]), diameter=size, fill=color)
        elif shape == 'rectangle':
            viz = imgviz.draw.rectangle(
                viz,
                aabb1=(xy[1] - size / 2, xy[0] - size / 2),
                aabb2=(xy[1] + size / 2, xy[0] + size / 2),
                fill=color,
            )

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('original')
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(122)
    plt.title('markers')
    plt.imshow(viz)
    plt.axis('off')

    out_file = osp.join(here, '.readme/draw.jpg')
    plt.savefig(
        out_file, bbox_inches='tight', transparent='True', pad_inches=0
    )
    plt.close()

    img = imgviz.io.imread(out_file)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
