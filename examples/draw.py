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
    yxs = [(265, 265), (265, 330), (320, 315), (350, 270), (350, 320)]

    # eye segment
    viz = imgviz.draw.line(viz, yx=[yxs[0], yxs[1]], fill=(255, 255, 255), width=5)
    # mouse segment
    viz = imgviz.draw.line(viz, yx=[yxs[3], yxs[4]], fill=(255, 255, 255), width=5)

    colors = imgviz.label_colormap(value=255)[1:]
    shapes = ["star", "ellipse", "rectangle", "circle", "triangle"]
    for yx, color, shape in zip(yxs, colors, shapes):
        size = 20
        if shape == "star":
            viz = imgviz.draw.star(
                viz, center=(yx[0], yx[1]), size=1.2 * size, fill=color
            )
        elif shape == "ellipse":
            viz = imgviz.draw.ellipse(
                viz,
                yx1=(yx[0] - 8, yx[1] - 16),
                yx2=(yx[0] + 8, yx[1] + 16),
                fill=color,
            )
        elif shape == "circle":
            viz = imgviz.draw.circle(
                viz, center=(yx[0], yx[1]), diameter=size, fill=color
            )
        elif shape == "triangle":
            viz = imgviz.draw.triangle(
                viz, center=(yx[0], yx[1]), size=size, fill=color
            )
        elif shape == "rectangle":
            viz = imgviz.draw.rectangle(
                viz,
                aabb1=(yx[0] - size / 2, yx[1] - size / 2),
                aabb2=(yx[0] + size / 2, yx[1] + size / 2),
                fill=color,
            )
        else:
            raise ValueError(f"unsupport shape: {shape}")

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
