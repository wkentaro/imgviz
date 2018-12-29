#!/bin/bash

set -e
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
