# Visualize optical flow
# =======================
# Code modified from:
# https://github.com/tomrunia/OpticalFlow_Visualization/blob/master/flow_vis.py
from __future__ import annotations

import typing

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


class Flow2Rgb:
    def __init__(self, max_norm: float | np.floating | None = None) -> None:
        self._max_norm: np.float32 | None = (
            None if max_norm is None else np.float32(max_norm)
        )

    @property
    def max_norm(self) -> np.float32 | None:
        return self._max_norm

    def __call__(self, flow_uv: NDArray[np.floating]) -> NDArray[np.uint8]:
        """Visualize optical flow.

        Args:
            flow_uv: Optical flow with shape (H, W, 2).

        Returns:
            RGB image with shape (H, W, 3).
        """
        flow_viz, max_norm = flow2rgb(flow_uv, max_norm=self._max_norm, return_max=True)
        if self._max_norm is None:
            self._max_norm = max_norm
        return flow_viz


@typing.overload
def flow2rgb(
    flow_uv: NDArray[np.floating],
    max_norm: float | np.floating | None = ...,
    return_max: typing.Literal[False] = ...,
) -> NDArray[np.uint8]: ...


@typing.overload
def flow2rgb(
    flow_uv: NDArray[np.floating],
    max_norm: float | np.floating | None = ...,
    return_max: typing.Literal[True] = ...,
) -> tuple[NDArray[np.uint8], np.float32]: ...


def flow2rgb(
    flow_uv: NDArray[np.floating],
    max_norm: float | np.floating | None = None,
    return_max: bool = False,
) -> NDArray[np.uint8] | tuple[NDArray[np.uint8], np.float32]:
    """Visualize optical flow.

    Args:
        flow_uv: Optical flow with shape (H, W, 2).
        max_norm: Maximum norm for normalization. If None, use the maximum norm
            in the flow_uv.
        return_max: Whether to return the maximum norm used for normalization.

    Returns:
        RGB image with shape (H, W, 3), or tuple of (flow_rgb, max_norm).
    """
    if flow_uv.ndim != 3:
        raise ValueError(f"flow must be 3 dimensional, but got {flow_uv.ndim}")
    if flow_uv.shape[2] != 2:
        raise ValueError(f"flow must have shape (H, W, 2), but got {flow_uv.shape}")
    if not np.issubdtype(flow_uv.dtype, np.floating):
        raise ValueError(f"flow dtype must be float, but got {flow_uv.dtype}")

    flow_uv = flow_uv.astype(np.float32)

    if max_norm is None:
        norm: NDArray[np.float32] = np.linalg.norm(flow_uv, axis=2)
        max_norm = norm.max()
    else:
        max_norm = np.float32(max_norm)

    eps: np.float32 = np.finfo(np.float32).eps
    flow_u: NDArray[np.float32] = flow_uv[:, :, 0] / (max_norm + eps)
    flow_v: NDArray[np.float32] = flow_uv[:, :, 1] / (max_norm + eps)

    flow_rgb: NDArray[np.uint8] = _flow_compute_color(flow_u, flow_v)
    if return_max:
        return flow_rgb, max_norm
    else:
        return flow_rgb
