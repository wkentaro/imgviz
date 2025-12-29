import argparse
import os.path as osp

import matplotlib.pyplot as plt

import imgviz

here = osp.dirname(osp.abspath(__file__))


def run_example(function):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save", action="store_true", help="save image")
    args = parser.parse_args()

    img = function()

    if args.save:
        out_file = osp.join(here, f"assets/{function.__name__}.jpg")
        imgviz.io.imsave(out_file, img)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
