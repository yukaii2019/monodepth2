
name=tum_test_6
data_path=/home/remote/ykhsieh/tum_monodepth2/monodepth2/tum_dataset
log_path=./log

if [ ! -d ${log_path} ]; then
mkdir -p ${log_path}
fi

train_sequences=(
    fr1_360 fr1_desk fr1_desk2 fr1_floor fr1_plant fr1_room fr1_rpy fr1_teddy fr1_xyz
    fr2_360_hemisphere fr2_360_kidnap fr2_coke fr2_desk fr2_desk_with_person fr2_dishes
    fr2_flowerbouquet fr2_flowerbouquet_brownbackground fr2_large_no_loop fr2_large_with_loop
    fr2_metallic_sphere fr2_metallic_sphere2 fr2_pioneer_360 fr2_pioneer_slam fr2_pioneer_slam2
    fr2_pioneer_slam3 fr2_rpy fr2_xyz
    fr3_cabinet fr3_large_cabinet fr3_long_office_household fr3_nostructure_notexture_far
    fr3_nostructure_notexture_near_withloop fr3_nostructure_texture_far fr3_nostructure_texture_near_withloop
    fr3_sitting_halfsphere fr3_sitting_rpy fr3_sitting_static fr3_sitting_xyz
    fr3_structure_notexture_far fr3_structure_notexture_near fr3_structure_texture_far
    fr3_structure_texture_near fr3_teddy
    fr3_walking_halfsphere fr3_walking_rpy fr3_walking_static fr3_walking_xyz
)

CUDA_VISIBLE_DEVICES=0 python3 train.py \
--model_name ${name} \
--num_epochs 30 \
--batch_size 5 \
--data_path ${data_path} \
--pose_model_type "separate_resnet" \
--height 480 \
--width 640 \
--log_dir ${log_path} \
--dataset tum \
--frame_ids 0 -3 3 \
--train_seq "${train_sequences[@]}" \