#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $HERE

# check python version
PYTHON_VERSION="$TRAVIS_PYTHON_VERSION"
if [ "$PYTHON_VERSION" = "" ]; then
  PYTHON_VERSION="3.6"
fi
PYTHON_VERSION32="$(echo $PYTHON_VERSION | cut -d '.' -f 1)"

# install anaconda
if [ ! -d .anaconda$PYTHON_VERSION32 ]; then
  curl -L https://github.com/wkentaro/dotfiles/raw/master/local/bin/install_anaconda$PYTHON_VERSION32.sh | bash -s .
  source .anaconda$PYTHON_VERSION32/bin/activate
  conda install -y python=$PYTHON_VERSION
fi
source .anaconda$PYTHON_VERSION32/bin/activate

set -x

# flake8
pip install -U flake8
flake8 .

# mypy
if [ "$PYTHON_VERSION32" = "3" ]; then
  pip install -U mypy
  mypy -p imgviz --ignore-missing-imports
fi

# install
if [ "$PYTHON_VERSION32" = "2" ]; then
  # numpy 1.16 raises error on python2 with import dask.array
  pip install 'numpy<1.16'
fi
pip install -e .[all]

# pytest
pip install -U pytest
pytest -v tests

# examples
MPLBACKEND=agg python getting_started.py
for f in examples/*.py; do
  MPLBACKEND=agg python $f
done
