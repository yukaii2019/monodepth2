#!/bin/bash
experiment=$1
seq=$2
mode=$3

bash evo_traj.sh $experiment $seq $mode
bash evo_rpe.sh $experiment $seq $mode 
bash evo_ape.sh $experiment $seq $mode 