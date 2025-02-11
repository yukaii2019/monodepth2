#!/bin/bash

# Rename folders starting with "rgbd_dataset_freiburg1_"
for dir in rgbd_dataset_freiburg1_*; do
  mv "$dir" "${dir/rgbd_dataset_freiburg1_/fr1_}"
done

# Rename folders starting with "rgbd_dataset_freiburg2_"
for dir in rgbd_dataset_freiburg2_*; do
  mv "$dir" "${dir/rgbd_dataset_freiburg2_/fr2_}"
done

# Rename folders starting with "rgbd_dataset_freiburg3_"
for dir in rgbd_dataset_freiburg3_*; do
  mv "$dir" "${dir/rgbd_dataset_freiburg3_/fr3_}"
done
