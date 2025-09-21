name=kitti_30_epoch
data_path=/home_backup/sshuang/KITTI_dataset/kitti_raw 
log_path=./log

if [ ! -d ${log_path} ]; then
mkdir -p ${log_path}
fi

CUDA_VISIBLE_DEVICES=1 python3 train.py \
--model_name ${name} \
--num_epochs 30 \
--batch_size 12 \
--data_path ${data_path} \
--pose_model_type "separate_resnet" \
--height 192 \
--width 640 \
--log_dir ${log_path} \
--dataset kitti \
--png \