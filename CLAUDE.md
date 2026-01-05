# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

imgviz is a Python library for image visualization, providing tools to colorize depth images, label maps, instance segmentation masks, optical flow, and more. It works with numpy arrays and outputs RGB images suitable for display or saving.

## Development Commands

```bash
# Install for development (using uv)
make setup

# Run all tests
make test

# Run a single test file
python -m pytest tests/_depth_test.py -v

# Run a specific test
python -m pytest tests/_depth_test.py::test_depth2rgb -v

# Linting, formatting, and type checking
make lint      # Check formatting, linting, and types (uses ty)
make format    # Auto-fix formatting issues

# Build distribution
make build
```

## Architecture

The library is organized into visualization functions that convert data to RGB images:

- **Core visualization functions** (`imgviz/_*.py`):
  - `depth2rgb()` / `Depth2Rgb` - Depth image colorization
  - `flow2rgb()` / `Flow2Rgb` - Optical flow visualization
  - `label2rgb()` - Label map colorization
  - `instances2rgb()` - Instance segmentation visualization
  - `mask2rgb()` - Binary mask visualization
  - `nchannel2rgb()` / `Nchannel2Rgb` - Multi-channel image visualization
  - `tile()` - Tile multiple images into a grid
  - `resize()`, `centerize()` - Image transformations
  - `normalize()` - Image normalization
- **Drawing primitives** (`imgviz/draw/`): Shapes (`circle`, `ellipse`, `rectangle`, `triangle`, `star`, `line`) and text functions. Uses `Ink` type alias for color parameters.
- **Color utilities** (`imgviz/_color.py`): Color space conversions (`rgb2gray`, `gray2rgb`, `hsv2rgb`, `rgb2hsv`, `asrgb`, `asrgba`, `asgray`)
- **Sample data** (`imgviz/data/`): Built-in datasets for examples and testing (`arc2017`, `middlebury`, `kitti_odometry`, `voc`, `lena`)
- **I/O** (`imgviz/io.py`): `imread()`, `imsave()`, `lblsave()` for image I/O

All public functions are exported from `imgviz/__init__.py`.

## Key Patterns

- Functions accept numpy arrays and return `NDArray[np.uint8]` RGB/RGBA images
- Visualization functions typically have a class version (e.g., `Depth2Rgb`, `Flow2Rgb`) for reuse with consistent parameters
- The `examples/` directory contains runnable scripts demonstrating each major feature
- Type hints are used throughout; `py.typed` marker enables PEP 561 support
- Draw functions have two variants: `func()` returns a new array, `func_()` modifies a PIL.Image in-place

## Important Notes

- **README.md is auto-generated** from `generate_readme.py`. Do not edit README.md directly; update `generate_readme.py` instead and regenerate.
- **Type annotations**: The library uses `NDArray[np.uint8]`, `NDArray[np.float32]`, etc. for precise typing. Data functions return `TypedDict` for better type inference.
