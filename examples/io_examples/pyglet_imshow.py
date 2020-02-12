#!/usr/bin/env python

import imgviz


def get_images():
    data = imgviz.data.arc2017()
    yield data["rgb"]
    yield imgviz.depth2rgb(data["depth"], min_value=0.3, max_value=1)
    yield imgviz.label2rgb(data["class_label"])


def main():
    imgviz.io.pyglet_imshow(next(get_images()), "ndarray")
    imgviz.io.pyglet_run()

    imgviz.io.pyglet_imshow(get_images(), "generator")
    imgviz.io.pyglet_run()

    imgviz.io.pyglet_imshow(list(get_images()), "list")
    imgviz.io.pyglet_run()


if __name__ == "__main__":
    main()
