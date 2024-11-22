#!/usr/bin/env python

import glob
import os.path as osp

import PIL.Image


def tabulate(rows):
    html = "<table>"
    for row in rows:
        html += "\n\t<tr>"
        for col in row:
            html += "\n\t\t<td>{}</td>".format(col)
        html += "\n\t</tr>"
    html += "\n</table>"
    return html


def _get_github_image_url(relpath: str) -> str:
    return f"https://github.com/wkentaro/imgviz/raw/main/{relpath}"


def main():
    examples = []
    for py_file in sorted(glob.glob("examples/*.py")):
        img_file = osp.splitext(osp.basename(py_file))[0] + ".jpg"
        img_file = osp.join("examples/.readme", img_file)
        if not osp.exists(img_file):
            continue
        img = PIL.Image.open(img_file)
        width = 20.0 / img.height * img.width
        examples.append(
            (
                f'<pre><a href="{py_file}">{py_file}</a></pre>',
                f'<img src="{_get_github_image_url(relpath=img_file)}" width="{width}%" />',  # NOQA: E501
            )
        )
    examples = tabulate(examples)

    # TODO: read from pyproject.toml
    dependencies = []
    for req in ["matplotlib", "numpy", "Pillow>=5.3.0", "PyYAML"]:
        pkg = req
        for sep in "<=>":
            pkg = pkg.split(sep)[0]
        dependencies.append("- [{0}](https://pypi.org/project/{1})".format(req, pkg))
    dependencies = "\n".join(dependencies)

    py_file = "getting_started.py"
    with open(py_file) as f:
        active = False
        lines = []
        for line in f:
            if line == "# GETTING_STARTED {{\n":
                active = True
                continue
            elif line == "# }} GETTING_STARTED\n":
                active = False
                continue
            if active:
                lines.append(line)
    getting_started = "".join(lines)

    README = """\
<!-- DO NOT EDIT THIS FILE MANUALLY. This file is generated by generate_readme.py. -->

<h1 align="center">
  imgviz
</h1>

<h4 align="center">
  Image Visualization Tools
</h4>

<div align="center">
  <a href="https://pypi.python.org/pypi/imgviz"><img src="https://img.shields.io/pypi/v/imgviz.svg"></a>
  <a href="https://pypi.org/project/imgviz"><img src="https://img.shields.io/pypi/pyversions/imgviz.svg"></a>
  <a href="https://github.com/wkentaro/imgviz/actions"><img src="https://github.com/wkentaro/imgviz/workflows/ci/badge.svg"></a>
</div>

<div align="center">
  <a href="#installation"><b>Installation</b></a> |
  <a href="#getting-started"><b>Getting Started</b></a> |
  <a href="#examples"><b>Examples</b></a> |
  <a href="https://github.com/wkentaro/imgviz-cpp"><b>C++ Version</b></a>
</div>

<br/>

<div align="center">
  <img src="https://github.com/wkentaro/imgviz/raw/main/.readme/getting_started.jpg" width="95%">
</div>

## Installation

```bash
pip install imgviz

# there are optional dependencies like skimage, below installs all.
pip install imgviz[all]
```


## Dependencies

{dependencies}

## Getting Started

```python
# getting_started.py

{getting_started}```

## [Examples](examples)

{examples}
"""  # NOQA

    README = README.format(
        getting_started=getting_started,
        dependencies=dependencies,
        examples=examples,
    )

    print(README, end="")


if __name__ == "__main__":
    main()
