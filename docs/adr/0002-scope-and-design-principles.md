# Scope and design principles

Status: accepted

`imgviz` is a programmatic compositor for scientific and technical imagery:
primitives, fills, layers, blend modes, final image. The mental model is
Affinity Designer or Figma as a Python library, reproducible and embeddable. It
stays small, composable, and framework-agnostic; every function is a pure
operation on NumPy arrays or PIL images. This ADR records what that commits us
to and what it rules out, so the same scope debates do not restart.

## In scope

- Composable visual primitives: shapes, masks, fills, text, images-as-layers
- Blend modes and alpha compositing as first-class operations
- Anti-aliased, publication-quality rendering
- Layout utilities: `pad`, `centerize`, `tile`, `resize`, `normalize`
- Color, dtype, and channel-order conversions handled correctly everywhere
- Minimal I/O for common image formats

## Out of scope

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
2. **Multi-channel and dtype-aware everywhere.** If a user would otherwise reach
   for `np.pad(img, ((t,b),(l,r),(0,0)), constant_values=...)`, `imgviz` should
   do it for them.
3. **Immutable by default.** Functional variants return new arrays; in-place
   variants are explicit and marked with a trailing underscore.
4. **Functions over frameworks.** Inputs and outputs are `numpy.ndarray` and
   `PIL.Image.Image`. Users handle tensor conversion at the boundary.

## Considered and rejected

| Idea                                               | Why rejected                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------- |
| COCO / YOLO / Pascal VOC / LabelMe parsers         | Application layer. Thin adapters belong downstream.                               |
| Keypoint schemas (COCO-17, MediaPipe-33, Halpe-26) | Application layer. The primitive takes points and edges; users supply the schema. |
| PyTorch / JAX / HuggingFace tensor interop         | Application layer. Users convert at the boundary.                                 |
| `compare()`, `tile_annotated()`                    | Trivially composable from existing `tile` and `text`.                             |
| `overlay()` as a separate function                 | Subsumed by `blend(mode="normal", alpha=...)`.                                    |
| `border()`                                         | Redundant with `pad` at equal sides.                                              |
| 3D box projection, BEV canvas                      | Application layer. Belongs in AV or point-cloud tooling.                          |
| GradCAM / SAM / attention-map helpers              | Compose from `blend` and mask primitives; no model-specific code in `imgviz`.     |
