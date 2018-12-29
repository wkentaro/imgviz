#!/bin/bash

set -e

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd $HERE

PYTHON_VERSION="$TRAVIS_PYTHON_VERSION"
if [ "$PYTHON_VERSION" = "" ]; then
  PYTHON_VERSION="3.6"
fi
PYTHON_VERSION32="$(echo $PYTHON_VERSION | cut -d '.' -f 1)"

if [ ! -d .anaconda$PYTHON_VERSION32 ]; then
  curl -L https://github.com/wkentaro/dotfiles/raw/master/local/bin/install_anaconda$PYTHON_VERSION32.sh | bash -s .
  source .anaconda$PYTHON_VERSION32/bin/activate
  conda install -y python=$PYTHON_VERSION
fi
source .anaconda$PYTHON_VERSION32/bin/activate

set -x

# flake8
pip install hacking
flake8 .

# mypy
if [ "$(python -c 'import sys; print(sys.version[0])')" = "3" ]; then
  pip install mypy
  mypy -p imgviz --ignore-missing-imports
fi

# install
pip install -e .

# pytest
pip install pytest
pytest -v tests

# examples
MPLBACKEND=agg python getting_started.py
for f in examples/*.py; do
  MPLBACKEND=agg python $f
done
