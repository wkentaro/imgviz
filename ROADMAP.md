# Roadmap

This document describes where `imgviz` is heading. It captures the project's
vision, what is and is not in scope, and the primitives planned for upcoming
releases. It also records ideas that were considered and explicitly rejected,
so the same discussions do not restart every six months.

## Vision

`imgviz` is a programmatic compositor for scientific and technical imagery —
primitives, fills, layers, blend modes, final image. The mental model is
Affinity Designer or Figma, but as a Python library: reproducible, scriptable,
embeddable in notebooks, pipelines, and batch jobs.

The library stays small, composable, and framework-agnostic. Every function is
a pure operation on NumPy arrays or PIL images.

## Scope

### In scope

- Composable visual primitives: shapes, masks, fills, text, images-as-layers
- Blend modes and alpha compositing as first-class operations
- Anti-aliased, publication-quality rendering
- Layout utilities: `pad`, `centerize`, `tile`, `resize`, `normalize`
- Color, dtype, and channel-order conversions handled correctly everywhere
- Minimal I/O for common image formats

### Out of scope

- Dataset format parsers (COCO, YOLO, Pascal VOC, LabelMe, CVAT)
- Model- or framework-specific adapters (YOLO, torchvision, detectron2, HuggingFace)
- Bundled keypoint or skeleton schemas (COCO-17, MediaPipe-33, Halpe-26)
- Interactive GUIs, annotation tools, video playback
- 3D rendering, BEV projection, point-cloud visualization

**The line:** if a feature requires knowledge of a specific dataset, model, or
downstream task, it belongs in a library that depends on `imgviz`, not in
`imgviz` itself.

## Design principles

1. **Only add primitives that are not trivially composable** from existing
   pieces. A three-line user script is not a feature request.
1. **Multi-channel and dtype-aware everywhere.** If a user would otherwise
   reach for `np.pad(img, ((t,b),(l,r),(0,0)), constant_values=...)`, `imgviz`
   should do it for them.
1. **Immutable by default.** Functional variants return new arrays; in-place
   variants are explicit and marked with a trailing underscore.
1. **Functions over frameworks.** Inputs and outputs are `numpy.ndarray` and
   `PIL.Image.Image`. Users handle tensor conversion at the boundary.

## Planned primitives

### Polygon and polyline

`draw.polygon`, `draw.polygon_`, `draw.polyline`, `draw.polyline_`. Accept
`(N, 2)` points. Integrate with the `Fill` system so any pattern can fill any
polygon. Closes [#92](https://github.com/wkentaro/imgviz/issues/92).

### Keypoints with edges

```python
draw.keypoints(image, points, edges=None, scores=None,
               point_colors=None, edge_colors=None)
```

Schema-free: the user supplies the edge list — pairs of indices. Optional
per-point `scores` in `[0, 1]` linearly modulate per-point alpha; an edge's
alpha is the minimum of its two endpoints. No COCO-17 or MediaPipe knowledge
lives in the library; users bring their own topology.

### More `Fill` patterns

Extends `Fill.Stripe` with `Checker`, `Dots`, `CrossHatch`, `Gradient` (linear
and radial), and `Noise`. Anywhere a drawer accepts a `Fill`, all patterns
work. Non-trivial to roll by hand (arbitrary-angle cross-hatch, radial
gradients) and universally useful.

### Extended draw family

- `draw.arrow` — line plus arrowhead with correct geometry at the tip.
- `draw.rotated_rectangle(center, size, angle)` — oriented bounding boxes.
- `draw.rounded_rectangle(yx1, yx2, radius)` — arc-cornered rectangles.
- `dash=(on, off)` style kwarg on stroke-based primitives: `line`,
  `polyline`, `polygon`, `rectangle`, `rotated_rectangle`,
  `rounded_rectangle`. Not applicable to fill-only primitives like `circle`.

### Blend modes

```python
imgviz.blend(a, b, mode="normal", alpha=1.0)
```

Ships with `mode="normal"` (linear alpha) — the workhorse "mask at 50%
opacity" case. Additional modes (`multiply`, `screen`, `overlay`, `darken`,
`lighten`, `add`, `difference`) land on demand as real use cases appear;
each is a one-function addition behind the same signature.

### Mask geometry

- `imgviz.mask.contour(mask)` — returns a list of `(N, 2)` polylines, one
  per connected region (a mask may have disjoint components).
- `imgviz.mask.convex_hull(mask)` — returns hull polygon.
- `imgviz.mask.outline(mask, width)` — returns an outline mask via dilation.

Composes with the polygon drawer and `Fill` patterns: trace a mask contour,
draw it as a `CrossHatch`-filled polygon with dashed stroke.

### Anti-aliased rendering

`antialias=True` kwarg on every `draw.*` primitive, implemented via
supersampling (SSAA). PIL's built-in AA is faster but inconsistent across
primitives and doesn't extend to mask outputs; SSAA gives one uniform
implementation that works for strokes, fills, and masks alike.
Publication-quality output without the user managing resolution manually.

### Rename `depth2rgb` → `colorize`

`depth2rgb` already applies a colormap to any 2D scalar field — depth maps,
attention maps, heatmaps, score fields, single-channel model outputs. The
name is the only barrier to discovery: a user with an attention map doesn't
think to look for `depth2rgb`.

```python
imgviz.colorize(scalar, vmin=None, vmax=None, cmap="viridis")
```

`depth2rgb` stays as a deprecated alias for one minor release before
removal in the next major version.

### Directional, multi-channel `pad`

```python
imgviz.pad(image, top=0, bottom=0, left=0, right=0, color=(0, 0, 0))
```

Handles `HW`, `HWC`, and `HWCA` layouts; accepts RGB-tuple fills. Replaces
awkward `np.pad(img, ((t,b),(l,r),(0,0)), constant_values=...)` calls. The
existing `centerize` and `tile` border logic refactor to delegate here, so
fill semantics and dtype handling live in one place.

## Structural changes under consideration

These are larger shifts that affect the public API. Neither is committed;
each would start as a prototype in `examples/` before any core change.

### `Layer` + `compose()`

A `Layer(rgba, blend_mode, opacity, mask)` type and an `imgviz.compose(layers)`
function. Lets users build stacks iteratively — change one layer's opacity,
re-render — without recomputing the whole pipeline. Existing convenience
functions like `label2rgb` and `instances2rgb` become compositions of
underlying layers.

Open question: does the ergonomic win justify a parallel API surface?
Promotion criterion: at least two existing convenience functions
(e.g. `label2rgb`, `instances2rgb`) re-expressed as `compose()` calls in
`examples/` with no loss of clarity or output fidelity.

### RGBA throughout

Primitives today return `uint8 HWC` RGB. Returning RGBA would let alpha
propagate naturally through `compose()` — a polygon's alpha is zero outside
the shape, text's alpha is glyph coverage, a mask's alpha is the mask itself.
Would be a deliberate, breaking change reserved for a future major version.

## Dropped ideas

These came up during brainstorming but are explicitly out of scope or
redundant. Recorded here to save future discussions.

| Idea | Why dropped |
|---|---|
| COCO / YOLO / Pascal VOC / LabelMe parsers | Application layer. Thin adapters belong downstream. |
| Keypoint schemas (COCO-17, MediaPipe-33, Halpe-26) | Application layer. The primitive takes points and edges; users supply the schema. |
| PyTorch / JAX / HuggingFace tensor interop | Application layer. Users convert at the boundary. |
| `compare()`, `tile_annotated()` | Trivially composable from existing `tile` and `text`. |
| `overlay()` as a separate function | Subsumed by `blend(mode="normal", alpha=...)`. |
| `border()` | Redundant with `pad` at equal sides. |
| 3D box projection, BEV canvas | Application layer. Belongs in AV or point-cloud tooling. |
| GradCAM / SAM / attention-map helpers | Compose from `blend` and mask primitives; no model-specific code in `imgviz`. |

## Proposing a new idea

Open a GitHub issue that states:

1. The primitive's name, signature, and input/output types.
1. One realistic use case that is *not* a three-line NumPy script.
1. Which design principle above the proposal satisfies.

Proposals that fit the scope and principles land on this roadmap. Proposals
that cross the scope line are closed with a link to this document.
