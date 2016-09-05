#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/catd_dogs

EXAMPLE=examples/cats_dogs
DATA=data/cats_dogs
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/train_lmdb \
  $DATA/mean.binaryproto

echo "Done."
