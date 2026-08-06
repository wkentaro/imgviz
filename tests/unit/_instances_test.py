import numpy as np
import pytest
from numpy.typing import NDArray

import imgviz
from imgviz._instances import masks_to_bboxes


def test_masks_to_bboxes() -> None:
    data = imgviz.data.arc2017()

    class_label = data["class_label"]
    masks = [class_label == label_id for label_id in np.unique(class_label)]
    bboxes = masks_to_bboxes(masks)

    assert len(bboxes) == len(masks)
    assert bboxes.shape[1] == 4

    ymin = bboxes[:, 0]
    xmin = bboxes[:, 1]
    ymax = bboxes[:, 2]
    xmax = bboxes[:, 3]
    height, width = class_label.shape
    assert ((0 <= ymin) & (ymin <= height - 1)).all()
    assert ((0 <= ymax) & (ymax <= height - 1)).all()
    assert ((0 <= xmin) & (xmin <= width - 1)).all()
    assert ((0 <= xmax) & (xmax <= width - 1)).all()


def test_masks_to_bboxes_leaves_empty_mask_as_zeros() -> None:
    masks = [np.zeros((10, 10), dtype=bool)]
    bboxes = masks_to_bboxes(masks)

    np.testing.assert_array_equal(bboxes, np.zeros((1, 4)))


@pytest.fixture
def image() -> NDArray[np.uint8]:
    return np.full((50, 50, 3), 30, dtype=np.uint8)


@pytest.fixture
def mask() -> NDArray[np.bool_]:
    mask = np.zeros((50, 50), dtype=bool)
    mask[10:40, 10:40] = True
    return mask


@pytest.fixture
def bbox() -> list[float]:
    return [10.0, 10.0, 39.0, 39.0]


def test_instances2rgb_draws_bboxes(
    image: NDArray[np.uint8], bbox: list[float]
) -> None:
    out = imgviz.instances2rgb(image=image, labels=[1], bboxes=[bbox])

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert (out != image).any()
    np.testing.assert_array_equal(out[25, 25], image[25, 25])


def test_instances2rgb_blends_masks(
    image: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> None:
    out = imgviz.instances2rgb(image=image, labels=[1], masks=[mask])

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out[25, 25], image[25, 25])
    np.testing.assert_array_equal(out[0, 0], image[0, 0])


def test_instances2rgb_rounds_blended_mask_colors() -> None:
    image = np.full((100, 100, 3), 50, dtype=np.uint8)
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:90] = True
    colormap = np.array([[0, 0, 0], [99, 99, 99]], dtype=np.uint8)

    out = imgviz.instances2rgb(
        image=image, labels=[0], masks=[mask], alpha=0.3, colormap=colormap
    )

    # 0.7 * 50 + 0.3 * 99 = 64.7 -> rounds to 65, not truncated to 64.
    np.testing.assert_array_equal(out[50, 50], [65, 65, 65])


def test_instances2rgb_draws_captions(
    image: NDArray[np.uint8], bbox: list[float]
) -> None:
    out = imgviz.instances2rgb(image=image, labels=[1], bboxes=[bbox], captions=["cat"])

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert (out != image).any()


def test_instances2rgb_draws_boundary(
    image: NDArray[np.uint8], mask: NDArray[np.bool_]
) -> None:
    out = imgviz.instances2rgb(image=image, labels=[1], masks=[mask], boundary_width=3)

    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert (out == (200, 200, 200)).all(axis=2).any()


def test_instances2rgb_runs_on_voc_example() -> None:
    data = imgviz.data.voc()
    captions = [data["class_names"][label_id] for label_id in data["labels"]]

    out = imgviz.instances2rgb(
        image=data["rgb"],
        bboxes=data["bboxes"],
        labels=data["labels"],
        captions=captions,
    )

    assert out.shape == data["rgb"].shape
    assert out.dtype == np.uint8


def test_instances2rgb_requires_bboxes_or_masks(
    image: NDArray[np.uint8],
) -> None:
    with pytest.raises(ValueError, match="bboxes or masks must be provided"):
        imgviz.instances2rgb(image=image, labels=[1])


def test_instances2rgb_rejects_negative_labels(
    image: NDArray[np.uint8], bbox: list[float]
) -> None:
    with pytest.raises(ValueError, match="all labels must be >= 0"):
        imgviz.instances2rgb(image=image, labels=[-1], bboxes=[bbox])


def test_instances2rgb_rejects_length_mismatch(
    image: NDArray[np.uint8], bbox: list[float]
) -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        imgviz.instances2rgb(image=image, labels=[1, 2], bboxes=[bbox])


def test_instances2rgb_rejects_non_uint8(
    image: NDArray[np.uint8], bbox: list[float]
) -> None:
    with pytest.raises(ValueError, match="image dtype must be np.uint8"):
        imgviz.instances2rgb(image=image.astype(float), labels=[1], bboxes=[bbox])


def test_instances2rgb_rejects_non_array(bbox: list[float]) -> None:
    with pytest.raises(TypeError, match="image must be a numpy array"):
        imgviz.instances2rgb(image=[[1]], labels=[1], bboxes=[bbox])  # type: ignore[arg-type]


def test_instances2rgb_accepts_grayscale_image(bbox: list[float]) -> None:
    gray = np.full((50, 50), 30, dtype=np.uint8)

    out = imgviz.instances2rgb(image=gray, labels=[1], bboxes=[bbox])

    assert out.shape == (50, 50, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], np.full(3, gray[0, 0]))


def test_instances2rgb_rejects_4d_image(bbox: list[float]) -> None:
    image = np.full((50, 50, 3, 1), 30, dtype=np.uint8)
    with pytest.raises(ValueError, match="image must be 2 or 3 dimensional"):
        imgviz.instances2rgb(image=image, labels=[1], bboxes=[bbox])


def test_instances2rgb_skips_empty_mask(image: NDArray[np.uint8]) -> None:
    empty_mask = np.zeros((50, 50), dtype=bool)

    out = imgviz.instances2rgb(image=image, labels=[1], masks=[empty_mask])

    np.testing.assert_array_equal(out, image)


def test_instances2rgb_skips_zero_area_bbox(image: NDArray[np.uint8]) -> None:
    zero_area_bbox = [10.0, 10.0, 10.0, 10.0]
    out = imgviz.instances2rgb(image=image, labels=[1], bboxes=[zero_area_bbox])

    np.testing.assert_array_equal(out, image)
