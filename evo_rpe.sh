#!/bin/bash
experiment=$1
seq=$2
mode=$3

if [ ! -d "./results/$experiment/$seq/$mode/rpe" ]; then
mkdir -p ./results/$experiment/$seq/$mode/rpe
fi

align_flags="--align_origin"

if [ "$mode" == "scale" ]; then
    align_flags="--align_origin --correct_scale"
elif [ "$mode" == "6dof" ]; then
    align_flags="--align"
elif [ "$mode" == "7dof" ]; then
    align_flags="--align --correct_scale"
fi


evo_rpe tum tum_dataset/$seq/groundtruth.txt ./results/$experiment/$seq/pred.txt \
--plot_mode=xyz \
--save_plot ./results/$experiment/$seq/$mode/rpe/ \
--save_result ./results/$experiment/$seq/$mode/rpe_result.zip \
$align_flags \



unzip -o ./results/$experiment/$seq/$mode/rpe_result.zip -d ./results/$experiment/$seq/$mode/rpe/