#!/bin/bash
experiment=$1
seq=$2
mode=$3

if [ ! -d "./results/$experiment/$seq/${mode}/traj" ]; then
mkdir -p ./results/$experiment/$seq/${mode}/traj
fi

align_flags="--align_origin"

if [ "$mode" == "scale" ]; then
    align_flags="--align_origin --correct_scale"
elif [ "$mode" == "6dof" ]; then
    align_flags="--align"
elif [ "$mode" == "7dof" ]; then
    align_flags="--align --correct_scale"
fi


evo_traj tum ./results/$experiment/$seq/pred.txt \
--ref tum_dataset/$seq/groundtruth.txt \
--plot_mode=xyz \
--save_plot ./results/$experiment/$seq/${mode}/traj/ \
$align_flags \