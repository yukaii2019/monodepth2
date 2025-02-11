#!/bin/bash
for file in *.tgz; do
  tar -xvzf "$file"
done
