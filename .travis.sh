#!/bin/bash

set -e
set -x

export MPLBACKEND='agg'

for f in examples/*.py; do
  python $f
done
