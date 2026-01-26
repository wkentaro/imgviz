# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About

imgviz is a Python library for image visualization, providing tools for colorizing depth images, labeling, instance segmentation visualization, optical flow, and drawing primitives.

## Development Commands

```bash
# Setup (installs all dev dependencies)
make setup

# Run tests (parallelized with pytest-xdist)
make test

# Run a single test file
uv run pytest tests/_label_test.py -v

# Run a specific test
uv run pytest tests/_label_test.py::test_label2rgb -v

# Run tests with image display (for debugging)
uv run pytest tests/_label_test.py --show

# Lint and type check
make lint

# Format code
make format
```

## Architecture

### Core Modules (`imgviz/`)

- **Color conversions** (`_color.py`): RGB/HSV/grayscale/RGBA conversions
- **Depth visualization** (`_depth.py`): `depth2rgb()` with configurable colormaps via cmap library
- **Label visualization** (`_label.py`): `label2rgb()` with legend support and VOC-style colormap
- **Instance visualization** (`_instances.py`): `instances2rgb()` for bounding boxes and masks
- **Flow visualization** (`_flow.py`): `flow2rgb()` for optical flow
- **N-channel visualization** (`_nchannel.py`): `nchannel2rgb()` for multi-channel data
- **Image utilities**: `centerize()`, `resize()`, `tile()`, `normalize()`

### Drawing Module (`imgviz/draw/`)

All drawing functions have two variants:
- Functional (returns new image): `circle()`, `rectangle()`, `text()`, etc.
- In-place (modifies PIL Image, ends with `_`): `circle_()`, `rectangle_()`, `text_()`, etc.

### Sample Data (`imgviz/data/`)

Provides sample datasets for examples: `arc2017()`, `voc()`, `lena()`, `middlebury()`, `kitti()`.

## Type Annotations

The codebase uses NumPy type annotations with `NDArray` from `numpy.typing`. When adding type hints:
- Use `NDArray[np.uint8]` for image arrays
- Use `NDArray[np.integer]` for label arrays
- Use `NDArray[np.floating]` for depth/flow arrays
- Use `NDArray[np.bool_]` for mask arrays

## Testing

Tests use pytest with a custom `--show` flag for visual debugging. Tests are in `tests/` with naming convention `_<module>_test.py`.
