#!/bin/bash
mode=$1

test_sequences=(
    fr1_desk fr1_desk2 fr1_room fr2_desk fr2_xyz fr3_long_office_household 
)
tops=(
    63 127 255 512 -1
)


for seq in "${test_sequences[@]}"; do
    bash eval.sh tum_test_6 $seq $mode
    for top in "${tops[@]}"; do
        if [ "$top" == "-1" ]; then
            bash eval.sh tum_test_6_dpdm_top_all $seq $mode
        else
            bash eval.sh tum_test_6_dpdm_top_${top} $seq $mode
        fi
    done
done
