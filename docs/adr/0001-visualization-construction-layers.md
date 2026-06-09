# Visualization-construction layers and namespace placement

Status: accepted

We organize imgviz's drawing and visualization surface into three layers and
place them by one rule: a building block you assemble gets a namespace; an
end-product verb you call stays flat at the top level.

- **Primitives** — `imgviz.draw.*`: one atomic draw op, semantics-free
  (`circle`, `polygon`, `pie`).
- **Components** — `imgviz.components.*`: reusable composites of primitives plus
  layout, still semantics-free (`legend`, `text_in_rectangle`).
- **Compositions** — flat top-level verbs that turn domain data into a finished
  visualization (`label2rgb`, `instances2rgb`, `mask2rgb`, `colorize`).

`draw` and `io` keep their verb-shaped names: both have direct precedent
(`skimage.draw` / `skimage.io`, `PIL.ImageDraw`) and read as domain nouns.

Color and dtype conversions (`gray2rgb`, `bool2ubyte`, ...) stay flat at the top
level for backward compatibility and call-site convenience. They are boundary
adapters, not assembly pieces, so the building-block rule does not pull them into
a namespace. If they are ever grouped, the namespace is `color` (matching
`skimage.color`), never the bare verb `convert`.

## Considered and rejected

- **Fold components into `draw`** (the way scikit-image houses everything
  drawable in one module). Rejected: it collapses the primitive/component
  distinction this ADR exists to draw.
- **A `components` name with imaging-domain precedent.** There is none: no NumPy
  imaging library has a reusable semantics-free composite tier, because none aim
  to be a layered compositor. We accept the borrowed term over forcing a worse
  imaging-native noun.
- **Namespacing conversions now** (e.g. `imgviz.color.gray2rgb`). Deferred:
  churn on the most-called functions for little gain; revisit only as part of a
  major-version reorg behind back-compat aliases.

## Consequences

The top level is intentionally a hybrid: flat end-product verbs alongside
namespaced building blocks. This diverges from scikit-image's uniformly
namespaced submodules (it is closer to OpenCV's flat surface), trading a small
discoverability cost (users must learn which names are flat) for call-site
brevity. The hybrid rule must be documented prominently.
