#!/usr/bin/env python

from pyglet_imshow import get_images

import imgviz


def main():
    viewer = imgviz.io.PygletThreadedImageViewer(play=True)
    for image in get_images():
        viewer.imshow(image)


if __name__ == "__main__":
    main()
