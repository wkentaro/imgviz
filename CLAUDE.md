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
python -m pytest tests/test_depth.py -v

# Run a specific test
python -m pytest tests/test_depth.py::test_depth2rgb -v

# Linting, formatting, and type checking
make lint      # Check formatting, linting, and types (uses ty)
make format    # Auto-fix formatting issues

# Build distribution
make build
```

## Architecture

The library is organized into visualization functions that convert data to RGB images:

- **Core visualization functions** (`imgviz/*.py`): Main API functions like `depth2rgb()`, `label2rgb()`, `instances2rgb()`, `flow2rgb()`, `tile()`, `resize()`, `centerize()`
- **Drawing primitives** (`imgviz/draw/`): Low-level drawing functions for shapes and text
- **Color utilities** (`imgviz/color.py`): Color space conversions (rgb2gray, hsv2rgb, etc.)
- **Sample data** (`imgviz/data/`): Built-in datasets for examples and testing (arc2017, kitti, voc, etc.)
- **I/O backends** (`imgviz/_io/`): Display backends (pyplot, opencv, pyglet)

All public functions are exported from `imgviz/__init__.py`.

## Key Patterns

- Functions accept numpy arrays and return numpy uint8 RGB/RGBA images
- Visualization functions typically have a class version (e.g., `Depth2RGB`) for reuse with consistent parameters
- The `examples/` directory contains runnable scripts demonstrating each major feature

## Important Notes

- **README.md is auto-generated** from `generate_readme.py`. Do not edit README.md directly; update `generate_readme.py` instead and regenerate.
