
name=tum_test_0
data_path=/home/pylin/Research/dataset
log_path=./log


if [ ! -d ${log_path} ]; then
mkdir -p ${log_path}
fi

CUDA_VISIBLE_DEVICES=0 python3 train.py \
--model_name ${name} \
--num_epochs 30 \
--batch_size 2 \
--data_path ${data_path} \
--pose_model_type "separate_resnet" \
--height 480 \
--width 640 \
--log_dir ${log_path} \
--dataset tum \