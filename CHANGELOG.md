# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added `outline` and `outline_width` parameters to `flags2rgb` to override the pie glyph outline color and width ([#229](https://github.com/wkentaro/imgviz/pull/229))
- Added `draw.progress_bar` primitive for a horizontal progress bar overlay ([#209](https://github.com/wkentaro/imgviz/pull/209))
- Added `tint` for a whole-image color wash ([#204](https://github.com/wkentaro/imgviz/pull/204))
- Added `heatmap` for visualizing a list of points as a heatmap ([#202](https://github.com/wkentaro/imgviz/pull/202))
- Added `colorblind` for simulating color-vision deficiency ([#199](https://github.com/wkentaro/imgviz/pull/199))
- Added `draw.box_corners` primitive for a corner-only bounding-box style ([#197](https://github.com/wkentaro/imgviz/pull/197))
- Added `draw.arrow` primitive with an arrowhead at the tip ([#193](https://github.com/wkentaro/imgviz/pull/193))
- Added `draw.rotated_rectangle` primitive ([#184](https://github.com/wkentaro/imgviz/pull/184))
- Added `draw.rounded_rectangle` primitive ([#176](https://github.com/wkentaro/imgviz/pull/176))

### Changed

- Documented that `label2rgb`, `instances2rgb`, and `flags2rgb` accept a grayscale `(H, W)` image in addition to `(H, W, 3)`, matching the input they already convert internally ([#232](https://github.com/wkentaro/imgviz/pull/232))
- Changed `rgb2hsv` and `hsv2rgb` to validate input shape and dtype and raise a clear `ValueError`, matching the other color converters, instead of surfacing a confusing error from Pillow ([#222](https://github.com/wkentaro/imgviz/pull/222))

### Fixed

- Fixed `float2ubyte` silently accepting `NaN`, which defeated the `[0, 1]` range guards (since `NaN` compares false against both bounds) and produced garbage output; it now raises a clear `ValueError` ([#249](https://github.com/wkentaro/imgviz/pull/249))
- Fixed `letterbox` returning the input array itself when the image already matches the target size, so mutating the result no longer corrupts the input ([#219](https://github.com/wkentaro/imgviz/pull/219))
- Fixed `draw.text_in_rectangle` filling the grown canvas with the background's red channel replicated across every channel instead of the full RGB color ([#224](https://github.com/wkentaro/imgviz/pull/224))
- Fixed `components.legend` truncating instead of rounding its translucent background wash, which biased the blended pixels down by one ([#223](https://github.com/wkentaro/imgviz/pull/223))

## [2.1.0] - 2026-06-10

### Added

- Added `flags2rgb` composition for visualizing per-instance boolean flags as pie glyphs with a legend ([#170](https://github.com/wkentaro/imgviz/pull/170))
- Added `diff` for image difference visualization with signed/abs/ssim modes ([#144](https://github.com/wkentaro/imgviz/pull/144))
- Added `blur` and `pixelate` with optional mask for redaction ([#138](https://github.com/wkentaro/imgviz/pull/138))
- Added `letterbox` primitive ([#139](https://github.com/wkentaro/imgviz/pull/139))
- Added `pie` drawing primitive ([#146](https://github.com/wkentaro/imgviz/pull/146))
- Added `polygon` drawing primitive ([#121](https://github.com/wkentaro/imgviz/pull/121))
- Added directional multi-channel `imgviz.pad` ([#167](https://github.com/wkentaro/imgviz/pull/167))

### Changed

- Changed shape primitives (`circle_`, `ellipse_`, `rectangle_`, `star_`, `triangle_`) to raise when called with neither `fill` nor `outline`, instead of silently returning the image unchanged ([#123](https://github.com/wkentaro/imgviz/pull/123))
- Changed `asrgb(rgba, copy=True)` to return an independent array for RGBA input instead of a view ([#172](https://github.com/wkentaro/imgviz/pull/172))

### Fixed

- Fixed `imread` file-handle leak: the file handle is now closed after reading ([#124](https://github.com/wkentaro/imgviz/pull/124))
- Fixed `normalize` min/max collapse after float32 cast: min and max are kept distinct after casting to float32 ([#143](https://github.com/wkentaro/imgviz/pull/143))
- Fixed `Nchannel2Rgb` cache comparison: avoids an ambiguous array comparison in the cache ([#169](https://github.com/wkentaro/imgviz/pull/169))
- Fixed `text_in_rectangle` left alignment: `lb`/`lb-` text now aligns to the rectangle's left edge ([#175](https://github.com/wkentaro/imgviz/pull/175))
- Fixed `flow2rgb` color-wheel overflow: prevents an out-of-bounds color-wheel index ([#173](https://github.com/wkentaro/imgviz/pull/173))

## [2.0.1] - 2026-05-12

### Fixed

- Fixed `label2rgb` crash when every pixel is `-1` (all-unlabeled input)

## [2.0.0] - 2026-01-28

### Added

- Added `mask2rgb()` function to fill mask regions with color, with pattern fill support via `imgviz.fill` module
- Added `Flow2Rgb` class for consistent optical flow visualization
- Added `py.typed` PEP 561 marker file for type checker support
- Added `loc` options `"rt"` (right-top) and `"lb"` (left-bottom) to `centerize()`
- Added `max_norm` argument to `flow2rgb()` for normalization control
- Added `copy` parameter to `asrgb()`
- Added `width` parameter to `star()` and `triangle()` draw functions
- Exposed `masks_to_bboxes()` function
- Added `pathlib.Path` support to `imread()` and `imsave()`
- Added comprehensive type hints, including `TypedDict` return types for data functions (`arc2017()`, `middlebury()`, `voc()`) and a `NamedTuple` return type for `text_in_rectangle_aabb()`

### Changed

- **Breaking:** Renamed parameter `img` to `image` and `src` to `image` in all function arguments
- **Breaking:** Renamed parameter `imgs` to `images` in `tile()`
- **Breaking:** Renamed parameters `aabb1`/`aabb2` to `yx1`/`yx2` in draw functions
- **Breaking:** Replaced `shape` parameter with `height`/`width` in `centerize()` and `row`/`col` in `tile()`
- **Breaking:** Renamed class `Depth2RGB` to `Depth2Rgb`
- **Breaking:** Renamed class `Nchannel2RGB` to `Nchannel2Rgb`
- **Breaking:** Renamed `pil_imshow()` to `imshow()`
- **Breaking:** Simplified `label_colormap()` by removing the `value` parameter
- **Breaking:** Reordered `nchannel2rgb()` parameters; `dtype` is now keyword-only
- Bumped minimum Pillow version to 10.0.0
- Replaced assertions with proper `ValueError` exceptions for input validation
- Improved error messages to use lowercase initial letters
- Prefixed internal modules with `_`
- Adopted Google-style docstrings in draw module

### Removed

- **Breaking:** Removed `plot_trajectory()` function (eliminated matplotlib dependency)
- **Breaking:** Removed OpenCV backend for I/O
- **Breaking:** Removed pyglet backend for I/O
- **Breaking:** Removed matplotlib-based I/O functions (`pyplot_imshow`, `pyplot_to_numpy`, etc.)
- Removed deprecated `mask_to_bbox()` (use `masks_to_bboxes()` instead)

### Fixed

- Fixed `asrgba()` to correctly handle 2D grayscale images
- Fixed `centerize()` to handle `cval=0` correctly
- Fixed `min_max_value` check in `Nchannel2Rgb`
- Fixed `label2rgb()` centroid selection to be deterministic
- Improved error messages when optional dependencies (skimage, sklearn) are missing

## [1.8.0] - 2025-12-29

### Added

- Added support for `rt` (right-top) and `lb` (left-bottom) legend locations in `label2rgb`
- Added `label2rgb` legend padding that scales relative to font size for better scaling
- Bundled DejaVuSansMono.ttf font for consistent text rendering
- Added type annotations across the codebase

### Changed

- **Breaking:** Dropped Python 3.9 support (minimum is now Python 3.10)
- Replaced matplotlib with cmap for depth colormap (lighter dependency)

### Removed

- Removed PyYAML dependency (use JSON for camera info data)

## [1.7.6] - 2024-11-22

### Added

- Added `imgviz.io.lblsave` ([#65](https://github.com/wkentaro/imgviz/pull/65))

### Changed

- Migrated to pyproject.toml ([#66](https://github.com/wkentaro/imgviz/pull/66))

## [1.7.5] - 2023-12-30

### Added

- Added `masks_to_bboxes` that returns tight bounding boxes ([#63](https://github.com/wkentaro/imgviz/pull/63))

### Fixed

- Fixed `draw.text.text_size` with newline ([#59](https://github.com/wkentaro/imgviz/pull/59))

## [1.7.4] - 2023-08-22

### Added

- Added `hook` argument to `pyglet_imshow`

### Changed

- **Breaking:** Dropped Python 2.7 support; Python >= 3.5 is now required

### Fixed

- Fixed `pyglet_imshow_list` to display `images[window.index]` correctly

## [1.7.3] - 2023-07-27

### Fixed

- Fixed window size calculation in `pyglet_imshow` to correctly fit images within screen bounds
- Updated pyglet imshow window caption to include current index and total count (e.g. `filename 1/3`)

## [1.7.2] - 2023-02-15

### Fixed

- Supported older matplotlib versions via `font.getsize` fallback
- Supported older matplotlib versions via `matplotlib.cm.get_cmap` fallback

## [1.7.1] - 2023-02-15

### Fixed

- Fixed `DeprecationWarning` from `matplotlib.cm.get_cmap` by using `matplotlib.colormaps[]`
- Fixed `DeprecationWarning` from `getbbox` by using `getsize()[2:]`
- Fixed `DeprecationWarning` from `PIL.Image.LINEAR` by using `PIL.Image.BILINEAR`

## [1.7.0] - 2023-02-08

### Removed

- Removed `PygletThreadedImageViewer`

### Fixed

- Fixed `window_width` and `window_height` determination

## [1.6.2] - 2022-11-21

### Fixed

- Fixed `pyglet.__version__` compatibility

## [1.6.1] - 2022-11-20

### Fixed

- Supported pyglet >= 2.0.0

## [1.6.0] - 2022-11-20

### Added

- Supported `bool` dtype in `label2rgb`
- Improved `pyglet_imshow`: fit window to image and support keymap

## [1.5.1] - 2022-08-29

### Added

- Supported custom colors in `depth2rgb`
- Added `opencv-python` to the `[all]` extra

### Changed

- Deferred `matplotlib.pyplot` import to reduce side effects at import time

### Removed

- Removed `np.random.seed` call from internals

## [1.5.0] - 2022-03-07

### Added

- Added in-place drawing functions (`draw.*_`) for faster (~10-100x) rendering

## [1.4.2] - 2022-03-03

### Added

- Added `draw.ellipse` function
- Added `draw.line` function
- Added `asrgba` color conversion function
- Added more imshow functions to `imgviz._io`

## [1.4.1] - 2021-10-21

### Fixed

- Supported old NumPy versions that do not have `unpackbits(bitorder=)`

## [1.4.0] - 2021-10-06

### Added

- Supported dict as `alpha` argument in `label2rgb`

### Changed

- Broadcast `alpha` of `label2rgb` to list
- Replaced for-loop with numpy operation in `label_colormap`

## [1.3.0] - 2021-09-06

### Added

- Added `bool2ubyte`, `float2ubyte`, `asrgb` utility functions
- Supported grayscale image as input to `instances2rgb`

### Changed

- Made `loc="rb"` the default in label/image placement
- Moved `dtype` argument to the last position in `depth2rgb` signature
- Renamed `img` argument to `image` in `label2rgb`

### Fixed

- Fixed deprecation around `collections.abc`

## [1.2.6] - 2021-03-26

### Added

- Supported negative value in `shape` argument to `tile()`

## [1.2.5] - 2021-02-05

### Added

- Supported `dict` as `label_names` in `label2rgb`

## [1.2.4] - 2021-01-24

### Added

- Added `loc` parameter to `imgviz.centerize`

## [1.2.3] - 2020-11-15

### Changed

- Supported `int` for `fill` and `outline` arguments

## [1.2.2] - 2020-08-02

### Removed

- Removed modifier check in `on_key_press` in `pyglet_imshow`

### Fixed

- Fixed deprecated matplotlib colormap

## [1.2.1] - 2020-06-29

### Removed

- Removed padding from instances2rgb

## [1.2.0] - 2020-06-26

### Added

- Added `camera_info` field to `imgviz.data.arc2017()` return value, with PyYAML added as a dependency

## [1.1.2] - 2020-06-22

### Fixed

- Fixed color box size in legend

## [1.1.1] - 2020-06-21

### Added

- Added `font_path` parameter to `text_in_rectangle`

### Fixed

- Fixed image copy after PIL resize by using `np.array` instead of direct assignment
- Fixed random state in `Nchannel2RGB` to use a static random state for PCA

## [1.1.0] - 2020-05-13

### Added

- Added `lt+`, `lb-`, `rt+`, `rb-` anchor options for `imgviz.draw.text_in_rectangle`

### Fixed

- Fixed confusion of `cval` and `border` parameters in `tile()`
- Fixed skipping of label names that are `None`
- Stopped drawing rectangle for border in `tile()`

## [1.0.0] - 2020-04-28

### Removed

- Dropped Python 2 support

## [0.12.3] - 2020-04-26

### Added

- Added `font_path` parameter to `draw`

### Fixed

- Fixed array writability issue by using array copy

## [0.12.2] - 2020-03-28

### Added

- Added `wait_key` function

### Changed

- Used `RandomState` instead of `random.seed` for reproducibility
- Specified `pyglet<=1.5.0` dependency constraint
- Various improvements on `instances2rgb`
- Supported float height and width
- Initialized window in the first `imshow` call
- Used DejaVuSansMono as fixed-width font

### Fixed

- Fixed Python 3 compatibility
- Fixed missing import

## [0.12.1] - 2020-02-13

### Changed

- Improved API of `PygletThreadedImageViewer`

### Fixed

- Fixed exception at import when pyglet is not available

## [0.12.0] - 2020-02-09

### Added

- Added `PygletThreadedImageViewer` for threaded image viewing via Pyglet

## [0.11.1] - 2020-02-08

### Fixed

- Created image folder in `imsave` if it does not exist

## [0.11.0] - 2020-02-07

### Added

- Added `imgviz.asgray` function

## [0.10.7] - 2020-02-01

### Fixed

- Used `np.array` for PIL.Image conversion to fix compatibility with newer PIL versions

## [0.10.6] - 2020-01-26

### Added

- Added `imgviz.io.pyplot_to_numpy` function
- Added `imgviz.data.voc` dataset loader

### Changed

- Imported matplotlib optionally (no longer a hard dependency at import time)

### Removed

- Removed dict wrapper from npz in `imgviz.data.arc2017`

## [0.10.5] - 2020-01-10

- Maintenance release (packaging, CI, or internal changes only).

## [0.10.4] - 2020-01-08

- Maintenance release (packaging, CI, or internal changes only).

## [0.10.3] - 2020-01-01

- Maintenance release (packaging, CI, or internal changes only).

## [0.10.2] - 2019-12-28

### Fixed

- Fixed Pillow compatibility for `draw.rectangle` with `width` argument

## [0.10.1] - 2019-12-03

- Maintenance release (packaging, CI, or internal changes only).

## [0.10.0-1] - 2019-12-03

- Maintenance release (packaging, CI, or internal changes only).

## [0.10.0] - 2019-12-03

### Added

- Added `interpolation` parameter to `centerize`

### Changed

- Replaced `instances2rgb` with `text_in_rectangle`

### Fixed

- Fixed `Nchannel2RGB` when `min_value` and `max_value` are not provided
- Skipped drawing invalid masks

## [0.9.0] - 2019-07-17

### Added

- Added `aabb1` and `aabb2` options to `text_in_rectangle`
- Added `draw.triangle`

### Changed

- Determined foreground color with `get_fg_color`

## [0.8.0] - 2019-07-11

### Added

- Added `colormap` argument to `instances2rgb`

### Removed

- Removed `n_labels` parameter from `label2rgb`

## [0.7.0] - 2019-07-02

### Added

- Added `draw.star` function
- Added `data.lena` dataset
- Added `draw.circle` function
- Supported matplotlib legend-style label names visualization

### Changed

- **Breaking:** Renamed `rectangle(color=...)` parameter to `rectangle(outline=...)`
- Supported int32 images in resize

## [0.6.2] - 2019-06-29

### Fixed

- Supported multi-line text in `text_size`

## [0.6.1] - 2019-05-07

### Fixed

- Added `allow_pickle=True` to `np.load` call to avoid error with NumPy >= 1.16.3

## [0.6.0] - 2019-04-28

### Added

- Added `value` argument to `label_colormap`
- Added `text_in_rectangle` function

### Fixed

- Fixed `Depth2RGB` to use consistent `min_value` and `max_value`
- Fixed `_get_tile_shape` tile layout computation

## [0.5.0] - 2019-03-27

### Added

- Supported generator and list input in `pyglet_imshow`

## [0.4.0] - 2019-03-24

### Added

- Added `img` parameter to `label2rgb`

### Fixed

- Fixed type annotation issues for mypy compatibility

## [0.3.0] - 2019-03-17

### Changed

- Respected image size in `imgviz.tile`

## [0.2.8] - 2019-03-17

### Added

- Supported `caption` parameter in `pyglet_imshow`

### Fixed

- Fixed `tile` to support tiling all gray images

## [0.2.7] - 2019-03-01

- Maintenance release (packaging, CI, or internal changes only).

## [0.2.6] - 2019-02-12

### Fixed

- Fixed RGBA image handling in tile

## [0.2.5] - 2019-01-29

- Maintenance release (packaging, CI, or internal changes only).

## [0.2.4] - 2019-01-21

- Maintenance release (packaging, CI, or internal changes only).

## [0.2.3] - 2019-01-21

### Fixed

- Fixed dtype of `Depth2RGB`

## [0.2.2] - 2019-01-20

### Fixed

- Fixed tile function to correctly copy list and cast tuple to list

## [0.2.1] - 2019-01-19

- Maintenance release (packaging, CI, or internal changes only).

## [0.2.0] - 2019-01-14

### Added

- Added `Depth2RGB` class for depth image visualization
- Added `nchannel2rgb` (renamed from `ndim2rgb`) for multi-channel image visualization

### Changed

- Renamed `ndim2rgb` to `nchannel2rgb`; `nchannel` parameter replaces former `ndim`
- Changed `Nchannel2RGB` to accept `dtype` instead of `shape`

### Fixed

- Fixed random state for PCA in `nchannel2rgb`
- Fixed resizing of float nchannel images with Pillow

## [0.1.16] - 2019-01-11

### Changed

- Made `scikit-image` import optional so imgviz can be used without it installed

## [0.1.15] - 2019-01-11

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.14] - 2019-01-11

### Changed

- Made scikit-image an optional dependency

## [0.1.13] - 2019-01-11

### Added

- Added `normalize` function
- Added `border_width` parameter to relevant function

## [0.1.12] - 2019-01-10

### Changed

- Updated colormap for instance visualization

## [0.1.11] - 2019-01-03

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.10] - 2019-01-02

### Added

- Added optical flow visualization (`flow2rgb`)
- Added mask visualization support to `instances2rgb`

### Changed

- **Breaking:** Renamed `position` parameter to `yx` in `imgviz.draw.text`

## [0.1.9] - 2018-12-30

### Added

- Added `instances2rgb` function for instance segmentation colorization

## [0.1.8] - 2018-12-30

### Added

- Added `cv_imshow` and `cv_waitkey` to imgviz
- Added `pyplot_fig2arr` to `imgviz.io`
- Added `imgviz.data.kitti`
- Added trajectory visualization support

## [0.1.7] - 2018-12-29

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.6] - 2018-12-29

### Added

- Added `io` module to imgviz
- Added pyglet to requirements

## [0.1.5] - 2018-12-29

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.4] - 2018-12-29

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.3] - 2018-12-29

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.2] - 2018-12-29

- Maintenance release (packaging, CI, or internal changes only).

## [0.1.1] - 2018-12-29

### Fixed

- Fixed tile size computation

## [0.1.0] - 2018-12-29

### Added

- Added `color.label2rgb` function
- Added `text` to `imgviz.draw` module
- Added `class_label` field to `data.arc2017`
- Added `class_names` to `data.arc2017`

### Changed

- Renamed `colorize.depth2rgb` to `color.depth2rgb`

### Removed

- Removed color example

## [0.0.9] - 2018-12-28

- Maintenance release (packaging, CI, or internal changes only).

## [0.0.8] - 2018-12-28

- Maintenance release (packaging, CI, or internal changes only).

## [0.0.7] - 2018-12-28

- Maintenance release (packaging, CI, or internal changes only).

## [0.0.6] - 2018-12-28

- Maintenance release (packaging, CI, or internal changes only).

## [0.0.5] - 2018-12-28

- Maintenance release (packaging, CI, or internal changes only).

## [0.0.4] - 2018-12-28

### Added

- Supported float values for `height` and `width` parameters

## [0.0.3] - 2018-12-28

### Added

- Added `imgviz.data` module

## [0.0.2] - 2018-12-27

### Added

- Added `tile` function
- Added `rgb2rgba` function
- Added `draw.rectangle` function
- Added `rgb2gray` and `gray2rgb` functions
- Added `centerize` function
- Added `resize` function

## [0.0.1] - 2018-12-26

### Added

- Added `depth2rgb` function to colorize depth arrays as RGB images using matplotlib colormaps

[0.0.1]: https://github.com/wkentaro/imgviz/releases/tag/v0.0.1
[0.0.2]: https://github.com/wkentaro/imgviz/compare/v0.0.1...v0.0.2
[0.0.3]: https://github.com/wkentaro/imgviz/compare/v0.0.2...v0.0.3
[0.0.4]: https://github.com/wkentaro/imgviz/compare/v0.0.3...v0.0.4
[0.0.5]: https://github.com/wkentaro/imgviz/compare/v0.0.4...v0.0.5
[0.0.6]: https://github.com/wkentaro/imgviz/compare/v0.0.5...v0.0.6
[0.0.7]: https://github.com/wkentaro/imgviz/compare/v0.0.6...v0.0.7
[0.0.8]: https://github.com/wkentaro/imgviz/compare/v0.0.7...v0.0.8
[0.0.9]: https://github.com/wkentaro/imgviz/compare/v0.0.8...v0.0.9
[0.1.0]: https://github.com/wkentaro/imgviz/compare/v0.0.9...v0.1.0
[0.1.1]: https://github.com/wkentaro/imgviz/compare/v0.1.0...v0.1.1
[0.1.10]: https://github.com/wkentaro/imgviz/compare/v0.1.9...v0.1.10
[0.1.11]: https://github.com/wkentaro/imgviz/compare/v0.1.10...v0.1.11
[0.1.12]: https://github.com/wkentaro/imgviz/compare/v0.1.11...v0.1.12
[0.1.13]: https://github.com/wkentaro/imgviz/compare/v0.1.12...v0.1.13
[0.1.14]: https://github.com/wkentaro/imgviz/compare/v0.1.13...v0.1.14
[0.1.15]: https://github.com/wkentaro/imgviz/compare/v0.1.14...v0.1.15
[0.1.16]: https://github.com/wkentaro/imgviz/compare/v0.1.15...v0.1.16
[0.1.2]: https://github.com/wkentaro/imgviz/compare/v0.1.1...v0.1.2
[0.1.3]: https://github.com/wkentaro/imgviz/compare/v0.1.2...v0.1.3
[0.1.4]: https://github.com/wkentaro/imgviz/compare/v0.1.3...v0.1.4
[0.1.5]: https://github.com/wkentaro/imgviz/compare/v0.1.4...v0.1.5
[0.1.6]: https://github.com/wkentaro/imgviz/compare/v0.1.5...v0.1.6
[0.1.7]: https://github.com/wkentaro/imgviz/compare/v0.1.6...v0.1.7
[0.1.8]: https://github.com/wkentaro/imgviz/compare/v0.1.7...v0.1.8
[0.1.9]: https://github.com/wkentaro/imgviz/compare/v0.1.8...v0.1.9
[0.10.0]: https://github.com/wkentaro/imgviz/compare/v0.9.0...v0.10.0
[0.10.0-1]: https://github.com/wkentaro/imgviz/compare/v0.10.0...v0.10.0-1
[0.10.1]: https://github.com/wkentaro/imgviz/compare/v0.10.0-1...v0.10.1
[0.10.2]: https://github.com/wkentaro/imgviz/compare/v0.10.1...v0.10.2
[0.10.3]: https://github.com/wkentaro/imgviz/compare/v0.10.2...v0.10.3
[0.10.4]: https://github.com/wkentaro/imgviz/compare/v0.10.3...v0.10.4
[0.10.5]: https://github.com/wkentaro/imgviz/compare/v0.10.4...v0.10.5
[0.10.6]: https://github.com/wkentaro/imgviz/compare/v0.10.5...v0.10.6
[0.10.7]: https://github.com/wkentaro/imgviz/compare/v0.10.6...v0.10.7
[0.11.0]: https://github.com/wkentaro/imgviz/compare/v0.10.7...v0.11.0
[0.11.1]: https://github.com/wkentaro/imgviz/compare/v0.11.0...v0.11.1
[0.12.0]: https://github.com/wkentaro/imgviz/compare/v0.11.1...v0.12.0
[0.12.1]: https://github.com/wkentaro/imgviz/compare/v0.12.0...v0.12.1
[0.12.2]: https://github.com/wkentaro/imgviz/compare/v0.12.1...v0.12.2
[0.12.3]: https://github.com/wkentaro/imgviz/compare/v0.12.2...v0.12.3
[0.2.0]: https://github.com/wkentaro/imgviz/compare/v0.1.16...v0.2.0
[0.2.1]: https://github.com/wkentaro/imgviz/compare/v0.2.0...v0.2.1
[0.2.2]: https://github.com/wkentaro/imgviz/compare/v0.2.1...v0.2.2
[0.2.3]: https://github.com/wkentaro/imgviz/compare/v0.2.2...v0.2.3
[0.2.4]: https://github.com/wkentaro/imgviz/compare/v0.2.3...v0.2.4
[0.2.5]: https://github.com/wkentaro/imgviz/compare/v0.2.4...v0.2.5
[0.2.6]: https://github.com/wkentaro/imgviz/compare/v0.2.5...v0.2.6
[0.2.7]: https://github.com/wkentaro/imgviz/compare/v0.2.6...v0.2.7
[0.2.8]: https://github.com/wkentaro/imgviz/compare/v0.2.7...v0.2.8
[0.3.0]: https://github.com/wkentaro/imgviz/compare/v0.2.8...v0.3.0
[0.4.0]: https://github.com/wkentaro/imgviz/compare/v0.3.0...v0.4.0
[0.5.0]: https://github.com/wkentaro/imgviz/compare/v0.4.0...v0.5.0
[0.6.0]: https://github.com/wkentaro/imgviz/compare/v0.5.0...v0.6.0
[0.6.1]: https://github.com/wkentaro/imgviz/compare/v0.6.0...v0.6.1
[0.6.2]: https://github.com/wkentaro/imgviz/compare/v0.6.1...v0.6.2
[0.7.0]: https://github.com/wkentaro/imgviz/compare/v0.6.2...v0.7.0
[0.8.0]: https://github.com/wkentaro/imgviz/compare/v0.7.0...v0.8.0
[0.9.0]: https://github.com/wkentaro/imgviz/compare/v0.8.0...v0.9.0
[1.0.0]: https://github.com/wkentaro/imgviz/compare/v0.12.3...v1.0.0
[1.1.0]: https://github.com/wkentaro/imgviz/compare/v1.0.0...v1.1.0
[1.1.1]: https://github.com/wkentaro/imgviz/compare/v1.1.0...v1.1.1
[1.1.2]: https://github.com/wkentaro/imgviz/compare/v1.1.1...v1.1.2
[1.2.0]: https://github.com/wkentaro/imgviz/compare/v1.1.2...v1.2.0
[1.2.1]: https://github.com/wkentaro/imgviz/compare/v1.2.0...v1.2.1
[1.2.2]: https://github.com/wkentaro/imgviz/compare/v1.2.1...v1.2.2
[1.2.3]: https://github.com/wkentaro/imgviz/compare/v1.2.2...v1.2.3
[1.2.4]: https://github.com/wkentaro/imgviz/compare/v1.2.3...v1.2.4
[1.2.5]: https://github.com/wkentaro/imgviz/compare/v1.2.4...v1.2.5
[1.2.6]: https://github.com/wkentaro/imgviz/compare/v1.2.5...v1.2.6
[1.3.0]: https://github.com/wkentaro/imgviz/compare/v1.2.6...v1.3.0
[1.4.0]: https://github.com/wkentaro/imgviz/compare/v1.3.0...v1.4.0
[1.4.1]: https://github.com/wkentaro/imgviz/compare/v1.4.0...v1.4.1
[1.4.2]: https://github.com/wkentaro/imgviz/compare/v1.4.1...v1.4.2
[1.5.0]: https://github.com/wkentaro/imgviz/compare/v1.4.2...v1.5.0
[1.5.1]: https://github.com/wkentaro/imgviz/compare/v1.5.0...v1.5.1
[1.6.0]: https://github.com/wkentaro/imgviz/compare/v1.5.1...v1.6.0
[1.6.1]: https://github.com/wkentaro/imgviz/compare/v1.6.0...v1.6.1
[1.6.2]: https://github.com/wkentaro/imgviz/compare/v1.6.1...v1.6.2
[1.7.0]: https://github.com/wkentaro/imgviz/compare/v1.6.2...v1.7.0
[1.7.1]: https://github.com/wkentaro/imgviz/compare/v1.7.0...v1.7.1
[1.7.2]: https://github.com/wkentaro/imgviz/compare/v1.7.1...v1.7.2
[1.7.3]: https://github.com/wkentaro/imgviz/compare/v1.7.2...v1.7.3
[1.7.4]: https://github.com/wkentaro/imgviz/compare/v1.7.3...v1.7.4
[1.7.5]: https://github.com/wkentaro/imgviz/compare/v1.7.4...v1.7.5
[1.7.6]: https://github.com/wkentaro/imgviz/compare/v1.7.5...v1.7.6
[1.8.0]: https://github.com/wkentaro/imgviz/compare/v1.7.6...v1.8.0
[2.0.0]: https://github.com/wkentaro/imgviz/compare/v1.8.0...v2.0.0
[2.0.1]: https://github.com/wkentaro/imgviz/compare/v2.0.0...v2.0.1
[2.1.0]: https://github.com/wkentaro/imgviz/compare/v2.0.1...v2.1.0
[unreleased]: https://github.com/wkentaro/imgviz/compare/v2.1.0...HEAD
