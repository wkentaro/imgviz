# imgviz

A programmatic compositor for scientific and technical imagery: build a final
image from layered visual pieces, as pure operations on NumPy arrays and PIL
images.

## Language

### Visualization-construction stack

**Primitive**:
A single atomic drawing operation that places one shape or mark on an image,
knowing nothing about what the data means. A circle, a line, a pie.
_Avoid_: shape, glyph, mark

**Component**:
A reusable visual assembled from primitives plus layout, still knowing nothing
about data semantics: it receives already-resolved colors and text, not domain
data. A legend, a colorbar.
_Avoid_: widget, element, part

**Composition**:
An end-to-end function that turns domain data into a finished visualization,
applying the data's semantics and assembling primitives and components to do it.
`label2rgb`, `instances2rgb`, `mask2rgb`.
_Avoid_: visualizer, recipe, "2rgb function"

### Building blocks & operations

**Fill**:
A pattern that colors the interior of a shape: solid, stripe, checker, gradient.
_Avoid_: pattern, texture, brush

**Blend**:
Alpha compositing of one image or layer over another.
_Avoid_: overlay, merge

**Layer**:
An image-as-element carrying its own opacity, mask, and blend mode, stacked into
a final image.
_Avoid_: plane, slot

**Colormap**:
A mapping from scalar values or integer ids to RGB colors.
_Avoid_: palette, LUT
