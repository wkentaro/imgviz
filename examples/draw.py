#!/usr/bin/env python

import matplotlib.pyplot as plt

import imgviz


def draw():
    img = imgviz.data.lena()
    H, W = img.shape[:2]
    viz = img

    y1, x1 = 200, 180
    y2, x2 = 400, 380
    viz = imgviz.draw.rectangle(
        viz, (y1, x1), (y2, x2), outline=(255, 255, 255), width=5
    )
    viz = imgviz.draw.text_in_rectangle(
        viz,
        loc="lt",
        text="face",
        size=30,
        background=(255, 255, 255),
        aabb1=(y1, x1),
        aabb2=(y2, x2),
    )

    # eye, eye, nose, mouse, mouse
    xys = [(265, 265), (330, 265), (315, 320), (270, 350), (320, 350)]
    colors = imgviz.label_colormap(value=255)[1:]
    shapes = ["star", "star", "rectangle", "circle", "triangle"]
    for xy, color, shape in zip(xys, colors, shapes):
        size = 20
        if shape == "star":
            viz = imgviz.draw.star(
                viz, center=(xy[1], xy[0]), size=1.2 * size, fill=color
            )
        elif shape == "circle":
            viz = imgviz.draw.circle(
                viz, center=(xy[1], xy[0]), diameter=size, fill=color
            )
        elif shape == "triangle":
            viz = imgviz.draw.triangle(
                viz, center=(xy[1], xy[0]), size=size, fill=color
            )
        elif shape == "rectangle":
            viz = imgviz.draw.rectangle(
                viz,
                aabb1=(xy[1] - size / 2, xy[0] - size / 2),
                aabb2=(xy[1] + size / 2, xy[0] + size / 2),
                fill=color,
            )
        else:
            raise ValueError("unsupport shape: {}".format(shape))

    img = imgviz.draw.text_in_rectangle(
        img,
        loc="lt+",
        text="original",
        size=30,
        background=(255, 255, 255),
    )
    viz = imgviz.draw.text_in_rectangle(
        viz,
        loc="lt+",
        text="markers",
        size=30,
        background=(255, 255, 255),
    )

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title("original")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(122)
    plt.title("markers")
    plt.imshow(viz)
    plt.axis("off")

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == "__main__":
    from base import run_example

    run_example(draw)
