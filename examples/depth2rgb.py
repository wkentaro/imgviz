import matplotlib.pyplot as plt

import imgviz


def depth2rgb():
    data = imgviz.data.arc2017()

    depthviz = imgviz.depth2rgb(data['depth'], min_value=0.3, max_value=1)

    # -------------------------------------------------------------------------

    plt.figure(dpi=200)

    plt.subplot(121)
    plt.title('rgb')
    plt.imshow(data['rgb'])
    plt.axis('off')

    plt.subplot(122)
    plt.title('depth (colorized)')
    plt.imshow(depthviz)
    plt.axis('off')

    img = imgviz.io.pyplot_to_numpy()
    plt.close()

    return img


if __name__ == '__main__':
    from base import run_example

    run_example(depth2rgb)
