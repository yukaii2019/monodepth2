#!/bin/bash
experiment=$1
seq=$2

if [ ! -d "./results/$experiment/$seq/traj" ]; then
mkdir -p ./results/$experiment/$seq/traj
fi


evo_traj tum ./results/$experiment/$seq/pred.txt --ref tum_dataset/$seq/groundtruth.txt --plot_mode=xyz --save_plot ./results/$experiment/$seq/traj/ --align_origin