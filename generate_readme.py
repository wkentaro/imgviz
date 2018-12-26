#!/usr/bin/env python

from __future__ import print_function

import glob
import os.path as osp


def tabulate(rows):
    html = '<table>'
    for row in rows:
        html += '\n\t<tr>'
        for col in row:
            html += '\n\t\t<td>{}</td>'.format(col)
        html += '\n\t</tr>'
    html += '\n</table>'
    return html


def main():
    examples = []
    for py_file in glob.glob('examples/*.py'):
        img_file = osp.splitext(osp.basename(py_file))[0] + '.jpg'
        img_file = osp.join('examples/.readme', img_file)
        if not osp.exists(img_file):
            continue
        examples.append((
            '<pre>{}</pre>'.format(py_file),
            '<img src="{}" height="200px" />'.format(img_file),
        ))
    examples = tabulate(examples)

    README = '''\
# imgviz: Image Visualization Tools

[![Build Status](https://travis-ci.com/wkentaro/imgviz.svg?branch=master)](https://travis-ci.com/wkentaro/imgviz)

## Installation

```bash
pip install imgviz
```

## [Examples](examples)

{examples}
'''.format(examples=examples)  # NOQA

    print(README, end='')


if __name__ == '__main__':
    main()
