#!/bin/bash
experiment=$1
seq=$2

python3 run_tum.py \
-s $seq \
--load_weights_folder ./log/$experiment/models/weights_29 \
--result_dir ./results/$experiment \
--frame_interval=1
