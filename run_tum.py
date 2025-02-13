import copy
import os
import json 
import sys
import argparse

from tqdm import tqdm
import cv2
import numpy as np
import PIL.Image as pil
import torch
from torch.utils.data import DataLoader

from datasets.tum_dataset import collect_image_paths
import datasets
import networks
from layers import transformation_from_parameters
from transformations import quaternion_from_matrix

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='TUM evaluate')
    parser.add_argument("-s", "--seq", type=str, default=None, help="tum sequence name")
    parser.add_argument("--load_weights_folder", type=str) 
    parser.add_argument("--result_dir", type=str)
    
    parser.add_argument("--frame_interval", type=int, default=1)
    parser.add_argument("--data_path", type=str, default="/home/remote/ykhsieh/tum_monodepth2/monodepth2/tum_dataset") 
    parser.add_argument("--height", type=int, default=480) 
    parser.add_argument("--width", type=int, default=640)

    
    args = parser.parse_args()

    device = torch.device('cuda')

    filenames, filenames_map = collect_image_paths(args.data_path, [args.seq], frame_interval=args.frame_interval, mode="test")
    

    dataset = datasets.TumDataset(args.data_path, 
                                  filenames, 
                                  args.height, 
                                  args.width,
                                  [0, args.frame_interval], 
                                  1, 
                                  is_train=False, 
                                  img_ext=None, 
                                  filenames_map=filenames_map)
            
    dataloader = DataLoader(dataset, 
                            16, 
                            shuffle=False,
                            num_workers=12, 
                            pin_memory=True, 
                            drop_last=False)
    

    pose_encoder_path = os.path.join(args.load_weights_folder, "pose_encoder.pth")
    pose_decoder_path = os.path.join(args.load_weights_folder, "pose.pth")

    pose_encoder = networks.ResnetEncoder(18, False, 2)
    pose_encoder.load_state_dict(torch.load(pose_encoder_path))

    pose_decoder = networks.PoseDecoder(pose_encoder.num_ch_enc, 1, 2)
    pose_decoder.load_state_dict(torch.load(pose_decoder_path))

    pose_encoder.eval().to(device)
    pose_decoder.eval().to(device)

    pred_poses = []

    print("-> Computing pose predictions")
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            posecnn_input = torch.cat([inputs[("color_aug", i, 0)] for i in [0, args.frame_interval]], 1).to(device)
            features = [pose_encoder(posecnn_input)]
            axisangle, translation = pose_decoder(features)
            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0]).cpu().numpy())

    # pred_poses = np.concatenate([np.eye(4).reshape(1, 4, 4)] + pred_poses)

    pred_poses = np.concatenate(pred_poses) 

    print("-> Computing global pose")
    
    global_poses = {}
    
    
    
    for i in range(0, len(filenames_map), args.frame_interval):
        time_stamp = os.path.splitext(filenames_map[(args.seq, i)])[0]
        
        if i != 0:
            rel_pose = pred_poses[i-args.frame_interval]
            cur_pose = ref_pose @ rel_pose 
        else:
            cur_pose = np.eye(4)

        global_poses[time_stamp] = cur_pose 
        ref_pose = cur_pose 



    # for i, (k, v) in tqdm(enumerate(filenames_map.items())):        
    #     time_stamp = os.path.splitext(v)[0]
    #     rel_pose = pred_poses[i]
    #     if i != 0:

    #         cur_pose = ref_pose @ rel_pose
    #     else:
    #         cur_pose = np.eye(4)
    #     global_poses[time_stamp] = cur_pose 
    #     ref_pose = cur_pose


    output_path = os.path.join(args.result_dir, args.seq)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    print("-> Change result to TUM format")
    result = []
    outputfile = open(os.path.join(output_path, "pred.txt"), "w")
    for time_stamp, pose in global_poses.items():
        orientation = quaternion_from_matrix(pose, False)
        position = pose[:3, 3]
        outputfile.write('{} {} {} {} {} {} {} {}\n'.format(time_stamp, position[0], position[1], position[2], 
                                             orientation[0], orientation[1], orientation[2], orientation[3]))

    outputfile.close()
        






    
