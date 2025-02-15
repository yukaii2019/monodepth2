#!/bin/bash

test_sequences=(
    fr1_desk fr1_desk2 fr1_room fr2_desk fr2_xyz fr3_long_office_household 
)
tops=(
    63 127 255 512 -1
)


for seq in "${test_sequences[@]}"; do
    bash run_tum.sh tum_test_6 $seq
    for top in "${tops[@]}"; do    
        bash run_tum_dpdm.sh tum_test_6 $seq $top
    done
done
