# Changelog fragments

Each user-facing change ships as one file here instead of editing
`CHANGELOG.md` directly, so concurrent pull requests never conflict.

Name the file `<PR number>.<type>.md`, where `<type>` is one of `added`,
`changed`, `deprecated`, `removed`, `fixed`, or `security`. A second fragment
for the same PR and type takes a counter suffix: `1234.fixed.2.md`. Add the
fragment after opening the PR, when its number is known.

Write the entry as a single line without a bullet or PR link; towncrier adds
the link from the filename. Prefix `**Breaking:**` for changes that bump the
major version.

Release a minor version for any ready backward-compatible improvement and a
patch for backward-compatible fixes; there is no minimum release size.

Development tasks require [just](https://just.systems/) and
[uv](https://docs.astral.sh/uv/). Run `just` to list the available recipes.

To release version `X.Y.Z`:

1. Run `just release X.Y.Z`.
2. Commit the updated changelog and deleted fragments, then tag that commit.
3. Create and publish the GitHub release for the tag.

Publishing the GitHub release publishes the package to PyPI.
