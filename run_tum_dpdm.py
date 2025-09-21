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
from transformations import quaternion_from_matrix, quaternion_matrix

from SuperGluePretrainedNetwork.models.superglue import SuperGlue
from SuperGluePretrainedNetwork.models.superpoint import SuperPoint
from utils import readTUMgt, findClosestTimestamp, draw_match_side
from frame_drawer import FrameDrawer

def se3_exp(x):
    '''
        implement exponential mapping of SE3
        the input is (6,) ndarray
        the output is the mapped transformation matrix
    '''

    lo  = x[:3] # related to translation
    phi = x[3:] # related to rotation    

    theta_square = phi[0] ** 2 + phi[1] ** 2 + phi[2] ** 2
    
    if theta_square == 0:
        # if theta = 0 => rot and J will be an identity matrix
        a1 = 0
        a2 = 0
        a3 = 0
        theta = 0
    else:
        theta = np.sqrt(theta_square)
        one_over_theta = 1/theta
        a1 = phi[0] * one_over_theta 
        a2 = phi[1] * one_over_theta
        a3 = phi[2] * one_over_theta


    sa = np.sin(theta)
    ca = np.cos(theta)
    C = 1 - ca

    rot = np.zeros((4, 4))

    rot[0, 0] = a1 * a1 * C + ca
    rot[0, 1] = a1 * a2 * C - a3 * sa
    rot[0, 2] = a3 * a1 * C + a2 * sa
    rot[1, 0] = a1 * a2 * C + a3 * sa
    rot[1, 1] = a2 * a2 * C + ca
    rot[1, 2] = a2 * a3 * C - a1 * sa
    rot[2, 0] = a3 * a1 * C - a2 * sa
    rot[2, 1] = a2 * a3 * C + a1 * sa
    rot[2, 2] = a3 * a3 * C + ca
    rot[3, 3] = 1


    J = np.zeros((3, 3))

    if theta == 0:
        J[0, 0] = 1.0
        J[0, 1] = 0.0
        J[0, 2] = 0.0
        J[1, 0] = 0.0
        J[1, 1] = 1.0
        J[1, 2] = 0.0
        J[2, 0] = 0.0
        J[2, 1] = 0.0
        J[2, 2] = 1.0
    else:
        sa_a = sa/theta
        one_sa_a = 1-sa_a
        one_ca_a = (1-ca)/theta
        J[0, 0] = a1 * a1 * one_sa_a + sa_a
        J[0, 1] = a1 * a2 * one_sa_a - a3 * one_ca_a
        J[0, 2] = a3 * a1 * one_sa_a + a2 * one_ca_a
        J[1, 0] = a1 * a2 * one_sa_a + a3 * one_ca_a
        J[1, 1] = a2 * a2 * one_sa_a + sa_a
        J[1, 2] = a2 * a3 * one_sa_a - a1 * one_ca_a
        J[2, 0] = a3 * a1 * one_sa_a - a2 * one_ca_a
        J[2, 1] = a2 * a3 * one_sa_a + a1 * one_ca_a
        J[2, 2] = a3 * a3 * one_sa_a + sa_a

    rot[:3, [3]] = J @ np.expand_dims(lo, -1)

    return rot

def BA_GN(p3d, p2d, K, it, pose):
    fx = K[0][0]   
    fy = K[1][1]    
    cx = K[0][2]    
    cy = K[1][2]  

    for i in range(it):
        H = np.zeros((6,6))
        b = np.zeros((6,))

        p3d_c = pose @ p3d.T
        z_inv = 1.0 / p3d_c[2]
        z2_inv = z_inv * z_inv    
        px = fx * p3d_c[0] / p3d_c[2] + cx
        py = fy * p3d_c[1] / p3d_c[2] + cy
        px = np.expand_dims(px,axis=0)
        py = np.expand_dims(py,axis=0)
        proj = np.concatenate((px,py))
        reproj_error = p2d.T - proj
        
        J = np.zeros((p2d.shape[0],2,6),dtype = np.float64)
        J[:,0,0] = -fx * z_inv
        J[:,0,1] = 0
        J[:,0,2] = fx * p3d_c[0] * z2_inv
        J[:,0,3] = fx * p3d_c[0] * p3d_c[1] * z2_inv
        J[:,0,4] = -fx - fx * p3d_c[0] * p3d_c[0] * z2_inv
        J[:,0,5] = fx * p3d_c[1] * z_inv
        J[:,1,0] = 0
        J[:,1,1] = -fy * z_inv
        J[:,1,2] = fy * p3d_c[1] * z2_inv
        J[:,1,3] = fy + fy * p3d_c[1] * p3d_c[1] * z2_inv
        J[:,1,4] = -fy * p3d_c[0] * p3d_c[1] * z2_inv
        J[:,1,5] = -fy * p3d_c[0] * z_inv
        
        reproj_error = np.expand_dims(reproj_error.T,axis=2)

        J_ = J.transpose((0,2,1))

        H = np.matmul(J_,J).sum(axis=0)
        b = -np.matmul(J_,reproj_error).sum(axis=0)

        dx = np.linalg.solve(H, b)

        pose = se3_exp(dx.squeeze()) @ pose

    return pose



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='TUM evaluate')
    parser.add_argument("-s", "--seq", type=str, default=None, help="tum sequence name")
    parser.add_argument("--load_weights_folder", type=str) 
    parser.add_argument("--result_dir", type=str)
    parser.add_argument("--top", type=int, default=-1, help="Set -1 to select all matches, otherwise specify a number.")

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

    superpoint = SuperPoint({})
    superpoint.eval().to(device)

    superglue = SuperGlue({'weights': 'indoor',})
    superglue.eval().to(device)
    
    K = dataset.K[:3,:3]
    K[0,0] *= args.width 
    K[0,2] *= args.width 
    K[1,1] *= args.height 
    K[1,2] *= args.height 


    pred_poses = []
    pred_keypoints = []
    pred_scores = []
    pred_descriptors = []

    

    print("-> Computing pose predictions")
    with torch.no_grad():
        for inputs in tqdm(dataloader):
            posecnn_input = torch.cat([inputs[("color_aug", i, 0)] for i in [0, args.frame_interval]], 1).to(device)
            features = [pose_encoder(posecnn_input)]
            axisangle, translation = pose_decoder(features)
            pred_poses.append(
                transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True).cpu().numpy())

            superpoint_input = inputs[("color_aug", 0, 0)]
            gray = (torch.tensor([0.2989, 0.5870, 0.1140]).reshape(1,3,1,1) * superpoint_input).sum(dim=1, keepdim=True).to(device)
            superpoint_output = superpoint({"image": gray})            
            pred_keypoints.extend(list(map(lambda t: t.detach().cpu(), superpoint_output["keypoints"])))
            pred_scores.extend(list(map(lambda t: t.detach().cpu(), superpoint_output["scores"])))
            pred_descriptors.extend(list(map(lambda t: t.detach().cpu(), superpoint_output["descriptors"])))
    
    # for the last images
    last_imgs = []
    for i in range(len(filenames_map)-args.frame_interval, len(filenames_map)):
        img = np.asarray(dataset.get_color(args.seq, i, "r", False)).transpose(2,0,1).astype(np.float32)/255
        last_imgs.append(img)
    last_imgs = np.stack(last_imgs)

    superpoint_input = torch.from_numpy(last_imgs)
    # superpoint_input = inputs[("color_aug", args.frame_interval, 0)]
    gray = (torch.tensor([0.2989, 0.5870, 0.1140]).reshape(1,3,1,1) * superpoint_input).sum(dim=1, keepdim=True).to(device)
    superpoint_output = superpoint({"image": gray})            
    pred_keypoints.extend(list(map(lambda t: t.detach().cpu(), superpoint_output["keypoints"])))
    pred_scores.extend(list(map(lambda t: t.detach().cpu(), superpoint_output["scores"])))
    pred_descriptors.extend(list(map(lambda t: t.detach().cpu(), superpoint_output["descriptors"]))) 
    pred_poses = np.concatenate(pred_poses) 

 
    # for debug. load gt pose directly
    '''
    pred_poses = []
    groundtruth = np.array(readTUMgt(os.path.join(args.data_path, args.seq, "groundtruth.txt")))
    groundtruth_timestamps = groundtruth[:, 0]
    groundtruth_pose = groundtruth[:, 1:] 
    for i in tqdm(range(len(filenames_map))):
        time_stamp = os.path.splitext(filenames_map[(args.seq, i)])[0]
        closest_idx = findClosestTimestamp(groundtruth_timestamps, float(time_stamp))
        xyz = groundtruth_pose[closest_idx][:3]
        quat = groundtruth_pose[closest_idx][3:]
        quat = np.roll(quat, 1, axis=0)  # shift 1 column -> w in front column
        groundtruth_pose_mat = quaternion_matrix(quat)
        groundtruth_pose_mat[:3, 3] = xyz 
        pred_poses.append(groundtruth_pose_mat)
    pred_poses = np.stack(pred_poses)
    '''
    

    # for debug. output gt pose directly
    '''
    global_poses = {}
    for i in tqdm(range(0, len(filenames_map), args.frame_interval)):
        time_stamp = os.path.splitext(filenames_map[(args.seq, i)])[0]
        if i != 0:
            rel_pose = np.linalg.inv(pred_poses[i-args.frame_interval]) @ pred_poses[i]
            cur_pose = ref_pose @ rel_pose 
        else:
            cur_pose = np.eye(4)
        global_poses[time_stamp] = cur_pose 
        ref_pose = cur_pose 
    '''
    drawer = FrameDrawer()

    print("-> Computing global pose") 
    global_poses = {}
    for i in tqdm(range(0, len(filenames_map), args.frame_interval)):
        time_stamp = os.path.splitext(filenames_map[(args.seq, i)])[0]
        if i != 0:
            # for debug. calculate relative pose from gt
            '''
            rel_pose = np.linalg.inv(pred_poses[i-args.frame_interval]) @ pred_poses[i]
            '''
            
            rel_pose = pred_poses[i-args.frame_interval]

            keypoint = pred_keypoints[i].unsqueeze(0)
            score = pred_scores[i].unsqueeze(0)
            descriptor = pred_descriptors[i].unsqueeze(0)

            superglue_output = superglue({"descriptors0": descriptor.to(device), 
                                          "descriptors1": ref_descriptor.to(device),
                                          "keypoints0": keypoint.to(device),
                                          "keypoints1": ref_keypoint.to(device), 
                                          "scores0": score.to(device), 
                                          "scores1": ref_score.to(device), 
                                          "image0": np.zeros((1, 1, args.height, args.width)),
                                          "image1": np.zeros((1, 1, args.height, args.width))
                                          })

            matches0 = superglue_output["matches0"][0].cpu().detach().numpy()
            mscores0 = superglue_output["matching_scores0"][0].cpu().detach().numpy()
            queryIdx = np.where(matches0 > -1)[0]
            trainIdx = matches0[queryIdx]

            if args.top == -1:  # If top is -1, select all matches
                idx = slice(None)
            else:
                conf = mscores0[queryIdx] 
                idx = (-conf).argsort()[:args.top]

            queryIdx = queryIdx[idx]
            trainIdx = trainIdx[idx]

            cur_points = keypoint[0][queryIdx].detach().cpu().numpy()
            ref_points = ref_keypoint[0][trainIdx].detach().cpu().numpy()
            
            cur_img = np.asarray(dataset.get_color(args.seq, i, "r", False))

            
            # out_img = draw_match_side(ref_img, ref_points, cur_img, cur_points, N=args.top, inliers=None)
            # cv2.imwrite("matching.png", out_img) 
            # exit()

            


            for itr in range(30):
                X = cv2.triangulatePoints(K @ rel_pose[:3], K @ np.eye(4)[:3], ref_points.astype(np.float64).T, cur_points.astype(np.float64).T)
                X /= X[3]
                rel_pose = BA_GN(X.T, ref_points, K, 5, rel_pose)

            cur_pose = ref_pose @ rel_pose
            drawer.main(i, cur_pose, None, ref_points, cur_points, ref_img, cur_img, inlier=None, N=args.top)

        else:
            cur_pose = np.eye(4)
            keypoint = pred_keypoints[0].unsqueeze(0)
            score = pred_scores[0].unsqueeze(0)
            descriptor = pred_descriptors[0].unsqueeze(0)
            cur_img = np.asarray(dataset.get_color(args.seq, 0, "r", False))


        global_poses[time_stamp] = cur_pose 
        

        ref_pose = cur_pose 
        ref_keypoint = keypoint
        ref_score = score
        ref_descriptor = descriptor
        ref_img = cur_img 



    output_path = os.path.join(args.result_dir, args.seq)
    if not os.path.exists(output_path):
        os.makedirs(output_path)


    print("-> Change result to TUM format")
    result = []
    outputfile = open(os.path.join(output_path, "pred.txt"), "w")
    for time_stamp, pose in global_poses.items():
        quat = quaternion_from_matrix(pose, False)
        quat = np.roll(quat, -1, axis=0) # shift -1 column -> w in back column
        position = pose[:3, 3]
        outputfile.write('{} {} {} {} {} {} {} {}\n'.format(time_stamp, position[0], position[1], position[2], 
                                             quat[0], quat[1], quat[2], quat[3]))

    outputfile.close()