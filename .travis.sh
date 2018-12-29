#!/bin/bash

set -e
set -x

pytest -v tests

python getting_started.py
for f in examples/*.py; do
  python $f
done
