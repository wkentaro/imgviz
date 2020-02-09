#!/usr/bin/env python

import imgviz

from check_pyglet_imshow import get_images


def main():
    viewer = imgviz.io.PygletThreadedImageViewer(play=True)
    for image in get_images():
        viewer.imshow(image)


if __name__ == '__main__':
    main()
