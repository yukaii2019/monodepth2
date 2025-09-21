#!/bin/bash
experiment=$1
seq=$2
top=$3

if [ "$top" == "-1" ]; then
    result_dir=./results/${experiment}_dpdm_top_all
else
    result_dir=./results/${experiment}_dpdm_top_${top}
fi



python3 run_tum_dpdm.py \
-s $seq \
--load_weights_folder ./log/$experiment/models/weights_19 \
--result_dir $result_dir \
--frame_interval=3 \
--top=$top
