# Visualize optical flow
# =======================
# Code modified from:
# https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _make_colorwheel() -> NDArray[np.floating]:
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def _flow_compute_color(flow_u: NDArray, flow_v: NDArray) -> NDArray[np.uint8]:
    H, W = flow_u.shape[:2]
    flow_image = np.zeros((H, W, 3), np.uint8)

    colorwheel = _make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]

    rad = np.sqrt(np.square(flow_u) + np.square(flow_v))
    a = np.arctan2(-flow_v, -flow_u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 1
    f = fk - k0

    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range?

        ch_idx = i
        flow_image[:, :, ch_idx] = np.floor(255 * col)

    return flow_image


def flow2rgb(flow_uv: NDArray) -> NDArray[np.uint8]:
    """Visualize optical flow.

    Parameters
    ----------
    flow_uv
        Optical flow with shape (H, W, 2).

    Returns
    -------
    rgb
        RGB image with shape (H, W, 3).

    """
    assert flow_uv.ndim == 3, "flow must be 3 dimensional"
    assert flow_uv.shape[2] == 2, "flow must have shape (H, W, 2)"
    assert np.issubdtype(flow_uv.dtype, np.floating), "float must be float type"

    flow_u = flow_uv[:, :, 0]
    flow_v = flow_uv[:, :, 1]

    rad = np.sqrt(np.square(flow_u) + np.square(flow_v))
    rad_max = np.max(rad)

    epsilon = 1e-5
    flow_u = flow_u / (rad_max + epsilon)
    flow_v = flow_v / (rad_max + epsilon)

    return _flow_compute_color(flow_u, flow_v)
