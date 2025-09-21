# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function
import os
import hashlib
import zipfile
from six.moves import urllib
import numpy as np
import cv2

def drawKeyPts(im,keyp,col,th):
    for curKey in keyp:
        x=np.int(curKey.pt[0])
        y=np.int(curKey.pt[1])
        size = np.int(curKey.size)
        cv2.circle(im,(x,y),size, col,thickness=th, lineType=8, shift=0) 
    return im    

def draw_match_side(img1, kp1, img2, kp2, N, inliers):
    """Draw matches on 2 sides

    Args:
        img1 (array, [HxWxC]): image 1
        kp1 (array, [Nx2]): keypoint for image 1
        img2 (array, [HxWxC]): image 2
        kp2 (array, [Nx2]): keypoint for image 2
        N (int): number of matches to be drawn
        inliers (array, [Nx1]): boolean mask for inlier
        
    Returns:
        out_img (array, [Hx2WxC]): output image with drawn matches
    """
    out_img = np.array([])
    
    # generate a list of keypoints to be drawn
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
                            dtype=np.int
                            )
    
    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx,_distance=0) for idx in range(len(cv_kp1))]

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

    # output_image = drawKeyPts(img1,cv_kp1,(0, 255, 255),5)
    # cv2.imwrite("kp.png", output_image) 

    # inlier/outlier plot option
    if inliers is not None:
        inlier_mask = inliers[kp_list].ravel().tolist()
        inlier_draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = inlier_mask, # draw only inliers
                    flags = 2)
        
        outlier_mask = (inliers==0)[kp_list].ravel().tolist()
        outlier_draw_params = dict(matchColor = (255,0,0), # draw matches in red color
                    singlePointColor = None,
                    matchesMask = outlier_mask, # draw only inliers
                    flags = 2)
        out_img1 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, outImg=out_img, **inlier_draw_params)
        out_img2 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, outImg=out_img, **outlier_draw_params)
        out_img = cv2.addWeighted(out_img1, 0.5, out_img2, 0.5, 0)
    else:
        out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)
    
    
    return out_img

def readTUMgt(file_path):
    temp_gt = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 8:
                    temp_gt.append(list(map(lambda t: float(t), parts)))
    return temp_gt 

def findClosestTimestamp(groundtruth_timestamps, rgb_timestamp):
    idx = np.searchsorted(groundtruth_timestamps, rgb_timestamp, side='left')
    if idx == 0:
        closest_idx = 0
    elif idx == len(groundtruth_timestamps):
        closest_idx = len(groundtruth_timestamps) - 1
    else:
        left_idx = idx - 1
        right_idx = idx
        closest_idx = left_idx if abs(groundtruth_timestamps[left_idx] - rgb_timestamp) < abs(groundtruth_timestamps[right_idx] - rgb_timestamp) else right_idx

    return closest_idx


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d


def sec_to_hm(t):
    """Convert time in seconds to time in hours, minutes and seconds
    e.g. 10239 -> (2, 50, 39)
    """
    t = int(t)
    s = t % 60
    t //= 60
    m = t % 60
    t //= 60
    return t, m, s


def sec_to_hm_str(t):
    """Convert time in seconds to a nice string
    e.g. 10239 -> '02h50m39s'
    """
    h, m, s = sec_to_hm(t)
    return "{:02d}h{:02d}m{:02d}s".format(h, m, s)


def download_model_if_doesnt_exist(model_name):
    """If pretrained kitti model doesn't exist, download and unzip it
    """
    # values are tuples of (<google cloud URL>, <md5 checksum>)
    download_paths = {
        "mono_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip",
             "a964b8356e08a02d009609d9e3928f7c"),
        "stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_640x192.zip",
             "3dfb76bcff0786e4ec07ac00f658dd07"),
        "mono+stereo_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip",
             "c024d69012485ed05d7eaa9617a96b81"),
        "mono_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_no_pt_640x192.zip",
             "9c2f071e35027c895a4728358ffc913a"),
        "stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_no_pt_640x192.zip",
             "41ec2de112905f85541ac33a854742d1"),
        "mono+stereo_no_pt_640x192":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_no_pt_640x192.zip",
             "46c3b824f541d143a45c37df65fbab0a"),
        "mono_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_1024x320.zip",
             "0ab0766efdfeea89a0d9ea8ba90e1e63"),
        "stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/stereo_1024x320.zip",
             "afc2f2126d70cf3fdf26b550898b501a"),
        "mono+stereo_1024x320":
            ("https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_1024x320.zip",
             "cdc5fc9b23513c07d5b19235d9ef08f7"),
        }

    if not os.path.exists("models"):
        os.makedirs("models")

    model_path = os.path.join("models", model_name)

    def check_file_matches_md5(checksum, fpath):
        if not os.path.exists(fpath):
            return False
        with open(fpath, 'rb') as f:
            current_md5checksum = hashlib.md5(f.read()).hexdigest()
        return current_md5checksum == checksum

    # see if we have the model already downloaded...
    if not os.path.exists(os.path.join(model_path, "encoder.pth")):

        model_url, required_md5checksum = download_paths[model_name]

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("-> Downloading pretrained model to {}".format(model_path + ".zip"))
            urllib.request.urlretrieve(model_url, model_path + ".zip")

        if not check_file_matches_md5(required_md5checksum, model_path + ".zip"):
            print("   Failed to download a file which matches the checksum - quitting")
            quit()

        print("   Unzipping model...")
        with zipfile.ZipFile(model_path + ".zip", 'r') as f:
            f.extractall(model_path)

        print("   Model unzipped to {}".format(model_path))
