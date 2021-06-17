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
  <a href="https://imgviz.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/imgviz/badge/?version=latest" alt="Documentation Status" /></a>
</div>

<div align="center">
  <a href="https://imgviz.readthedocs.io/en/latest/?badge=latest"><b>Documentation</b></a> |
  <a href="#installation"><b>Installation</b></a> |
  <a href="#getting-started"><b>Getting Started</b></a> |
  <a href="#examples"><b>Examples</b></a> |
  <a href="https://github.com/wkentaro/imgviz-cpp"><b>C++ Version</b></a>
</div>

<br/>

<div align="center">
  <img src=".readme/getting_started.jpg" width="95%">
</div>

## Installation

```bash
pip install imgviz

# there are optional dependencies like skimage, below installs all.
pip install imgviz[all]
```


## Dependencies

- [matplotlib](https://pypi.org/project/matplotlib)
- [numpy](https://pypi.org/project/numpy)
- [Pillow>=5.3.0](https://pypi.org/project/Pillow)
- [PyYAML](https://pypi.org/project/PyYAML)

## Getting Started

```python
# getting_started.py

import imgviz


# sample data of rgb, depth, class label and instance masks
data = imgviz.data.arc2017()

rgb = data["rgb"]
gray = imgviz.rgb2gray(rgb)

# colorize depth image with JET colormap
depth = data["depth"]
depthviz = imgviz.depth2rgb(depth, min_value=0.3, max_value=1)

# colorize label image
class_label = data["class_label"]
labelviz = imgviz.label2rgb(class_label, image=gray, label_names=data["class_names"], font_size=20)

# instance bboxes
bboxes = data["bboxes"].astype(int)
labels = data["labels"]
masks = data["masks"] == 1
captions = [data["class_names"][l] for l in labels]
maskviz = imgviz.instances2rgb(gray, masks=masks, labels=labels, captions=captions)

# tile instance masks
insviz = [(rgb * m[:, :, None])[b[0] : b[2], b[1] : b[3]] for b, m in zip(bboxes, masks)]
insviz = imgviz.tile(imgs=insviz, border=(255, 255, 255))
insviz = imgviz.resize(insviz, height=rgb.shape[0])

# tile visualization
tiled = imgviz.tile(
    [rgb, depthviz, labelviz, maskviz, insviz],
    shape=(1, 5),
    border=(255, 255, 255),
    border_width=5,
)
```

## [Examples](examples)

<table>
	<tr>
		<td><pre><a href="examples/centerize.py">examples/centerize.py</a></pre></td>
		<td><img src="examples/.readme/centerize.jpg" width="53.333333333333336%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/depth2rgb.py">examples/depth2rgb.py</a></pre></td>
		<td><img src="examples/.readme/depth2rgb.jpg" width="52.21052631578947%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/draw.py">examples/draw.py</a></pre></td>
		<td><img src="examples/.readme/draw.jpg" width="37.79047619047619%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/flow2rgb.py">examples/flow2rgb.py</a></pre></td>
		<td><img src="examples/.readme/flow2rgb.jpg" width="52.21052631578947%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/instances2rgb.py">examples/instances2rgb.py</a></pre></td>
		<td><img src="examples/.readme/instances2rgb.jpg" width="66.35451505016722%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/label2rgb.py">examples/label2rgb.py</a></pre></td>
		<td><img src="examples/.readme/label2rgb.jpg" width="76.01532567049807%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/nchannel2rgb.py">examples/nchannel2rgb.py</a></pre></td>
		<td><img src="examples/.readme/nchannel2rgb.jpg" width="52.21052631578947%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/plot_trajectory.py">examples/plot_trajectory.py</a></pre></td>
		<td><img src="examples/.readme/plot_trajectory.jpg" width="26.86868686868687%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/resize.py">examples/resize.py</a></pre></td>
		<td><img src="examples/.readme/resize.jpg" width="47.238095238095234%" /></td>
	</tr>
	<tr>
		<td><pre><a href="examples/tile.py">examples/tile.py</a></pre></td>
		<td><img src="examples/.readme/tile.jpg" width="35.812274368231044%" /></td>
	</tr>
</table>
