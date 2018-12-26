#!/bin/bash

set -e

export MPLBACKEND='agg'

set -x

for f in examples/*.py; do
  python $f
done
