import argparse
import io
import os.path as osp
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
from numpy.typing import NDArray

import imgviz

here = osp.dirname(osp.abspath(__file__))


class _ExampleFn(Protocol):
    __name__: str

    def __call__(self) -> None: ...


def _pyplot_to_numpy() -> NDArray[np.uint8]:
    f = io.BytesIO()
    plt.savefig(
        f,
        bbox_inches="tight",
        transparent=True,
        pad_inches=0,
        format="jpeg",
    )
    plt.close()
    f.seek(0)
    arr = np.asarray(PIL.Image.open(f))
    return arr


def run_example(example_fn: _ExampleFn) -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--save", action="store_true", help="save image")
    args = parser.parse_args()

    example_fn()

    img: NDArray[np.uint8] = _pyplot_to_numpy()

    if args.save:
        out_file = osp.join(here, f"assets/{example_fn.__name__}.jpg")
        imgviz.io.imsave(out_file, img)

    plt.imshow(img)
    plt.axis("off")
    plt.show()
