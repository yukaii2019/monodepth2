import cv2
import matplotlib as mpl
import numpy as np
import os
from easydict import EasyDict as edict

# def draw_match_temporal(img1, kp1, img2, kp2, N):
#     """Draw matches defined by kp1, kp2. Lay the matches on img2.

#     Args:
#         img1 (array, [HxWxC]): image 1
#         kp1 (array, [Nx2]): keypoint for image 1
#         img2 (array, [HxWxC]): image 2
#         kp2 (array, [Nx2]): keypoint for image 2
#         N (int): number of matches to be drawn
    
#     Returns:
#         out_img (array, [HxWxC]): output image with drawn matches
#     """
#     # initialize output image
#     out_img = img2.copy()

#     # generate a list of keypoints to be drawn
#     kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N,
#                                             dtype=np.int
#                                             )
#     for i in kp_list:
#         # get location of of keypoints
#         center1 = (kp1[i][0].astype(np.int), kp1[i][1].astype(np.int))
#         center2 = (kp2[i][0].astype(np.int), kp2[i][1].astype(np.int))

#         # randomly pick a color for the match
#         color = np.random.randint(0, 255, 3)
#         color = tuple([int(i) for i in color])

#         # draw line between keypoints
#         cv2.line(out_img, center1, center2, color, 2)
#     return out_img

def draw_match_side(img1, kp1, img2, kp2, inlier, N):
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
    kp_list = np.linspace(0, min(kp1.shape[0], kp2.shape[0])-1, N, dtype=np.int)
    

    # Convert keypoints to cv2.Keypoint object
    cv_kp1 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp1[kp_list]]
    cv_kp2 = [cv2.KeyPoint(x=pt[0], y=pt[1], _size=1) for pt in kp2[kp_list]]

    good_matches = [cv2.DMatch(_imgIdx=0, _queryIdx=idx, _trainIdx=idx, _distance=0) for idx in range(len(cv_kp1))]


    if inlier is not None:
        inlier_mask = inlier[kp_list].ravel().tolist()

        inlier_draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = inlier_mask, # draw only inliers
                    flags = 2)
        
        outlier_mask = (inlier==0)[kp_list].ravel().tolist()
        outlier_draw_params = dict(matchColor = (255,0,0), # draw matches in red color
                    singlePointColor = None,
                    matchesMask = outlier_mask, # draw only inliers
                    flags = 2)


        out_img1 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, outImg=out_img, **inlier_draw_params)
        out_img2 = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, outImg=out_img, **outlier_draw_params)
        out_img = cv2.addWeighted(out_img1, 0.5, out_img2, 0.5, 0)
    else:
        out_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches1to2=good_matches, outImg=out_img)
    
    # cv2.imwrite('./test/outlier/{:04d}_{:02d}.png'.format(i, itr), out_img)
    
    return out_img

# def load_poses_from_txt(file_name):
#         f = open(file_name, 'r')
#         s = f.readlines()
#         f.close()
#         poses = []
#         for cnt, line in enumerate(s):
#             P = np.eye(4)
#             # line_split = [float(i) for i in line.split(" ") if i != "" ]
#             line_split = [float(i) for i in line.split()]
#             withIdx = len(line_split) == 13
#             for row in range(3):
#                 for col in range(4):
#                     P[row, col] = line_split[row*4 + col + withIdx]
#             if withIdx:
#                 frame_idx = line_split[0]
#             else:
#                 frame_idx = cnt
#             poses.append(P)
#         return np.array(poses)



class FrameDrawer():
    """Frame drawer to display different visualizations
    """
    
    def __init__(self):
        """
        Args:
            cfg (edict): visualization configuration
        """
        # intialize drawer size

        cfg = {
            "enable": False,
            "save_img": True,
            "window_h": 900,
            "window_w": 1500,

            "kp_match":{
                "kp_num": 100,
                "vis_temp":{
                    "enable": False
                },
                "vis_side":{
                    "enable": True,
                }
            },

            "trajectory":{
                "vis_traj": False,
                "vis_gt_traj": True,
                "mono_scale": 1,
                "vis_scale": 1
            },
        }

        self.cfg = edict(cfg)

        self.h = self.cfg.window_h
        self.w = self.cfg.window_w
        self.img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        
        # visualization setting
        self.vis_scale = self.cfg.trajectory.vis_scale
        self.draw_scale = 1
        self.text_y = 0.9 # size ratio on y-direction that start drawing text
        
        # initialize data and data assignment
        self.data = {}
        self.display = {}
        self.initialize_drawer()
    
    def initialize_drawer(self):
        """Initialize drawer by assigning items to the drawer
        """
        h = self.h
        w = self.w

        # assign data to the layout
        drawer_layout = {
            'traj':  ([int(h/4*0), int(w/4*0)], [int(h/4*4*self.text_y), int(w/4*2)]),
            # 'match_temp':       ([int(h/4*0), int(w/4*2)], [int(h/4*1), int(w/4*4)]),
            'match_side':       ([int(h/4*1), int(w/4*2)], [int(h/4*2), int(w/4*4)]),
            'text': ([int(h/4*4*self.text_y) , int(w/4*0)], [int(h/4*4), int(w/4*2)]),
            # 'frame':            ([int(h/4*1), int(w/4*2)], [int(h/4*2), int(w/4*4)]),
            # 'depth':            ([int(h/4*2), int(w/4*2)], [int(h/4*3), int(w/4*3)]),
            # 'flow1':            ([int(h/4*2), int(w/4*3)], [int(h/4*3), int(w/4*4)]),
            # 'flow2':            ([int(h/4*3), int(w/4*2)], [int(h/4*4), int(w/4*3)]),
            # 'rigid_flow_diff':  ([int(h/4*3), int(w/4*2)], [int(h/4*4), int(w/4*3)]),
            # 'opt_flow_diff':    ([int(h/4*3), int(w/4*3)], [int(h/4*4), int(w/4*4)]),

            # # (Experiement Ver. only)
            # 'warp_diff':        ([int(h/4*3), int(w/4*2)], [int(h/4*4), int(w/4*3)]),
        }

        for key, locs in drawer_layout.items():
            self.assign_data(
                    item=key,
                    top_left=locs[0],
                    bottom_right=locs[1]
            )
        
        # initialize start point
        self.traj_y0 = int((drawer_layout['traj'][1][0] - drawer_layout['traj'][0][0])/2)
        # self.traj_y0 = int((drawer_layout['traj'][1][0] * self.text_y - drawer_layout['traj'][0][0])/2)
        self.traj_x0 = int((drawer_layout['traj'][1][1] - drawer_layout['traj'][0][1])/2)

    def assign_data(self, item, top_left, bottom_right):
        """assign data to the drawer image
        
        Args:
            item (str): item name
            top_left (list): [y, x] position of top left corner
            bottom_right (list): [y, x] position of bottom right corner
        """
        self.data[item] = self.img[
                                    top_left[0]:bottom_right[0],
                                    top_left[1]:bottom_right[1]
                                    ]
        self.display[item] = True


    def update_data(self, item, data):
        """update drawer content
        
        Args:
            item (str): item to be updated
            data (array, [HxWx3]): content to be updated, RGB format
        """
        data_bgr = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        vis_h, vis_w, _ = self.data[item].shape
        self.data[item][...] = cv2.resize(data_bgr, (vis_w, vis_h))


    def interface(self):
        key = cv2.waitKey(10) or 0xff

        # pause
        if key == ord('p'):
            print("Paused.")
            while True:
                key2 = cv2.waitKey(1) or 0xff

                # # Match_temp
                # if key2 == ord('1'):
                #     self.display['match_temp'] = not(self.display['match_temp'])
                #     print("Match(1): {}".format(self.display['match_temp']))

                # Match side
                if key2 == ord('2'):
                    self.display['match_side'] = not(self.display['match_side'])
                    print("Match(2): {}".format(self.display['match_side']))

                # # depth
                # if key2 == ord('3'):
                #     self.display['depth'] = not(self.display['depth'])
                #     print("depth: {}".format(self.display['depth']))

                # if key2 == ord('4'):
                #     self.display['flow1'] = not(self.display['flow1'])
                #     self.display['flow2'] = not(self.display['flow2'])
                #     self.display['opt_flow_diff'] = not(self.display['opt_flow_diff'])
                #     print("flow: {}".format(self.display['flow1']))

                # Continue
                if key2 == ord('c'):
                    print("Continue.")
                    return

        # # Match_temp
        # if key == ord('1'):
        #     self.display['match_temp'] = not(self.display['match_temp'])
        #     print("Match(1): {}".format(self.display['match_temp']))

        # # Match side
        # if key == ord('2'):
        #     self.display['match_side'] = not(self.display['match_side'])
        #     print("Match(2): {}".format(self.display['match_side']))

        # # depth
        # if key == ord('3'):
        #     self.display['depth'] = not(self.display['depth'])
        #     print("depth: {}".format(self.display['depth']))

        # # flow
        # if key == ord('4'):
        #     self.display['flow1'] = not(self.display['flow1'])
        #     self.display['flow2'] = not(self.display['flow2'])
        #     self.display['opt_flow_diff'] = not(self.display['opt_flow_diff'])
        #     print("flow: {}".format(self.display['flow1']))


    # def draw_traj(self, frame_idx, pred_pose, gt_pose):
    #     """draw trajectory and related information

    #     Args:
    #         pred_poses (dict): predicted poses w.r.t world coordinate system
    #         gt_poses (dict): predicted poses w.r.t world coordinate system
    #     """
    #     traj_map = self.data["traj"][:int(self.h * self.text_y)]
    #     traj_map_h, traj_map_w, _ = traj_map.shape



    #     # latest_id = max(pred_poses.keys())

    #     # draw scales
    #     mono_scale = self.cfg.trajectory.mono_scale    # scaling factor to align with gt (if gt is available)
    #     pred_draw_scale = self.draw_scale * mono_scale * self.vis_scale


    #     # Get predicted location
    #     cur_t = pred_pose[:3, 3]
    #     x, y, z = cur_t[0], cur_t[1], cur_t[2]
        
        
    #     draw_x =   int(x*pred_draw_scale) + self.traj_x0
    #     draw_y = -(int(z*pred_draw_scale)) + self.traj_y0




    #     if (draw_x > traj_map_w-0) or (draw_x < 0) or (draw_y > traj_map_h-0) or (draw_y < 0):
    #         # resize current visualiation
    #         scale = 0.9
    #         zoom_traj_map = cv2.resize(traj_map, (int(traj_map_w*scale), int(traj_map_h*scale)))
    #         zoom_h, zoom_w, _ = zoom_traj_map.shape

    #         # fit resized visulization into the Frame drawer
    #         traj_map[...] = 0
    #         top_left = [int(self.traj_y0 - zoom_h/2), int(self.traj_x0 - zoom_w/2)]
    #         bottom_right = [int(self.traj_y0 + zoom_h/2), int(self.traj_x0 + zoom_w/2)]
    #         traj_map[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]] = zoom_traj_map
            
    #         # draw new point
    #         self.draw_scale *= scale
    #         pred_draw_scale = self.draw_scale * mono_scale
    #         draw_x = int(round(x * pred_draw_scale))+ self.traj_x0
    #         draw_y = int(round(- (z * pred_draw_scale))) + self.traj_y0
    #         cv2.circle(traj_map, (draw_x , draw_y), 1, (0, 255, 0), max(1, int(10* self.draw_scale)))

    #         if self.cfg.trajectory.vis_gt_traj:
    #             gt_draw_scale = self.vis_scale * self.draw_scale
    #             gt_t = gt_pose[:3, 3]
    #             gt_x, gt_y, gt_z = gt_t[0], gt_t[1], gt_t[2]
    #             gt_draw_x = int(gt_x * gt_draw_scale) + self.traj_x0
    #             gt_draw_y = -(int(gt_z * gt_draw_scale)) + self.traj_y0
    #             cv2.circle(traj_map, (gt_draw_x , gt_draw_y), 1, (0, 0, 255), max(1, int(10* self.draw_scale)))
    #     else:
    #         # draw point
    #         cv2.circle(traj_map, (draw_x , draw_y), 1, (0, 255, 0), max(1, int(10* self.draw_scale)))

    #         # # draw GT
    #         if self.cfg.trajectory.vis_gt_traj:
    #             gt_draw_scale = self.vis_scale * self.draw_scale
    #             gt_t = gt_pose[:3, 3]
    #             gt_x, gt_y, gt_z = gt_t[0], gt_t[1], gt_t[2]
    #             gt_draw_x = int(gt_x * gt_draw_scale) + self.traj_x0
    #             gt_draw_y = -(int(gt_z * gt_draw_scale)) + self.traj_y0
    #             cv2.circle(traj_map, (gt_draw_x , gt_draw_y), 1, (0, 0, 255), max(1, int(10* self.draw_scale)))
        
    #     # draw origin
    #     cv2.circle(self.img, (self.traj_x0 , self.traj_y0), 1, (255, 255, 255), 10)

    #     ''' Draw text information '''
    #     traj = self.data["traj"]
    #     traj_h, traj_w, _ = traj.shape

    #     # create empty text block
    #     cv2.rectangle(traj, (0, int(traj_h*self.text_y)), (traj_w, traj_h), (0, 0, 0), -1)

    #     # frame
    #     text = "Frame index = {}".format(frame_idx)
    #     cv2.putText(traj, text, (int(traj_w * 0.01), int(traj_h * 0.92)),
    #                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    #     # # coordinate
    #     # text = "Coordinates: x={:.2f} y={:.2f} z={:.2f}".format(x, y, z)
    #     # cv2.putText(traj, text, (int(traj_w * 0.01), int(traj_h * 0.92)),
    #     #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        
    #     # # tracking mode
    #     # # text = "Tracking mode: {}".format(tracking_mode)
    #     # text = "Seq: 09"
    #     # cv2.putText(traj, text, (int(traj_w * 0.01), int(traj_h * 0.92)),
    #     #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    #     # text = "GT: red"
    #     # cv2.putText(traj, text, (int(traj_w * 0.01), int(traj_h * 0.96)),
    #     #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
    #     # text = "Estimated: green"
    #     # cv2.putText(traj, text, (int(traj_w * 0.01), int(traj_h * 0.99)),
    #     #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
        
    #     # # text = "p/c: pause/continue;  [1, 2, 3, 4]: enable/disable vis."
    #     # # cv2.putText(traj, text, (int(traj_w * 0.01), int(traj_h * 0.99)),
    #     # #             cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

    # def draw_match_temp(self, vo):
    #     """Draw optical flow vectors on image

    #     Args:
    #         vo (DFVO): DF-VO
    #     """
    #     if self.display['match_temp']:
    #         # check if data available
    #         if vo.cur_data.get(vo.cfg.visualization.kp_src, -1) is -1: return

    #         # Set number of kp to be visualized
    #         vis_kp_num = min(vo.cur_data[vo.cfg.visualization.kp_src].shape[0], 
    #                         vo.cfg.visualization.kp_match.kp_num)

    #         # Set keypoints
    #         vis_kp_ref = vo.ref_data[vo.cfg.visualization.kp_src]
    #         vis_kp_cur = vo.cur_data[vo.cfg.visualization.kp_src]

    #         # get image
    #         vis_match_temp = draw_match_temporal(
    #                 img1=vo.ref_data['img'],
    #                 kp1=vis_kp_ref,
    #                 img2=vo.cur_data['img'],
    #                 kp2=vis_kp_cur,
    #                 N=vis_kp_num
    #                 )
            
    #         # draw image
    #         self.update_data("match_temp", vis_match_temp)
    #     else:
    #         h, w, c = self.data["match_temp"][...].shape
    #         self.data["match_temp"][...] = np.zeros((h,w,c))

    def draw_match_side(self, ref_2d, cur_2d, ref_img, cur_img, inlier, N):
        """Draw optical flow vectors on image
        """
        
        if self.display['match_side']: 
            vis_match_side = draw_match_side(
                    img1=ref_img,
                    img2=cur_img,
                    kp1=ref_2d,
                    kp2=cur_2d,
                    inlier=inlier,
                    N=N
                )

            self.update_data("match_side", vis_match_side)

        else:
            h, w, c = self.data["match_side"][...].shape
            self.data["match_side"][...] = np.zeros((h,w,c))


    # def draw_frame(self, img):
    #     if self.display['frame']: 
    #         self.update_data("frame", img)

    #     else:
    #         h, w, c = self.data["frame"][...].shape
    #         self.data["frame"][...] = np.zeros((h,w,c))


    # def draw_depth(self, vo):
    #     """Draw depth/disparity map

    #     Args:
    #         vo (DFVO): DF-VO
    #     """
    #     if self.display['depth']:
    #         # choose depth source
    #         if vo.cfg.visualization.depth.use_tracking_depth:
    #             if vo.cur_data.get('depth', -1) is -1: return
    #             tmp_vis_depth = vo.cur_data['depth']
    #         else:
    #             if vo.cur_data.get('raw_depth', -1) is -1: return
    #             tmp_vis_depth = vo.cur_data['raw_depth']
            
    #         # visualize depth
    #         if vo.cfg.visualization.depth.depth_disp == 'depth':
    #             vis_depth = tmp_vis_depth
    #             normalizer = mpl.colors.Normalize(vmin=0, vmax=vo.cfg.depth.max_depth)
    #             mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
    #             colormapped_im = (mapper.to_rgba(vis_depth)[:, :, :3] * 255).astype(np.uint8)
    #             self.update_data("depth", colormapped_im)
            
    #         # visualize inverse depth
    #         if vo.cfg.visualization.depth.depth_disp == 'disp':
    #             vis_depth = 1 / (tmp_vis_depth+1e-3)
    #             vis_depth[tmp_vis_depth==0] = 0
    #             vmax = np.percentile(vis_depth, 90)
    #             normalizer = mpl.colors.Normalize(vmin=0, vmax=vmax)
    #             mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='magma')
    #             colormapped_im = (mapper.to_rgba(vis_depth)[:, :, :3] * 255).astype(np.uint8)
    #             self.update_data("depth", colormapped_im)
    #     else:
    #         h, w, c = self.data["depth"][...].shape
    #         self.data["depth"][...] = np.zeros((h,w,c))


    # def draw_flow(self, flow_data, flow_name):
    #     """Draw optical flow map 

    #     Args:
    #         flow_data (array, [2xHxW]): flow data
    #         flow_name (str): flow name
    #     """
    #     if self.display[flow_name]:
    #         vis_flow = flow_data.transpose(1,2,0)
    #         vis_flow = flow_to_image(vis_flow)
    #         self.update_data(flow_name, vis_flow)
    #     else:
    #         h, w, c = self.data[flow_name][...].shape
    #         self.data[flow_name][...] = np.zeros((h,w,c))

    # def draw_flow_consistency(self, vo):
    #     """Draw forward-backward flow consistency map

    #     Args:
    #         vo (DFVO): DF-VO
    #     """
    #     # check if data is available
    #     if vo.cur_data.get('fb_flow_mask', -1) is -1: return

    #     # set vmax for different score method
    #     if vo.cfg.kp_selection.local_bestN.enable and \
    #         vo.cfg.kp_selection.local_bestN.score_method == "flow_ratio":
    #             vmax = 0.1
    #     else:
    #         vmax = 1
        
    #     normalizer = mpl.colors.Normalize(vmin=0, vmax=vmax)
    #     mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='jet')
    #     mask = vo.cur_data['fb_flow_mask']
    #     colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
    #     self.update_data("opt_flow_diff", colormapped_im)

    # def draw_warp_diff(self, vo):
    #     """Draw warp diff 

    #     Args:
    #         vo (DFVO): DF-VO
    #     """
    #     # set vmax for different score method
    #     vmax = 1
        
    #     normalizer = mpl.colors.Normalize(vmin=0, vmax=vmax)
    #     mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='jet')
    #     mask = vo.deep_models.depth.warp_diff
    #     colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
    #     self.update_data("warp_diff", colormapped_im)

    # def draw_rigid_flow_consistency(self, vo):
    #     """Draw optical-rigid flow consistency map

    #     Args:
    #         vo (DFVO): DF-VO
    #     """
    #     # check if data is available
    #     if vo.cur_data.get('rigid_flow_mask', -1) is -1: return
        
    #     vmax = vo.cfg.kp_selection.rigid_flow_kp.rigid_flow_thre
    #     normalizer = mpl.colors.Normalize(vmin=0, vmax=vmax)
    #     mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='jet')
    #     mask = vo.cur_data['rigid_flow_mask']
    #     colormapped_im = (mapper.to_rgba(mask)[:, :, :3] * 255).astype(np.uint8)
    #     self.update_data("rigid_flow_diff", colormapped_im)

    def draw_text(self, frame_idx):

        text_img = self.data["text"]
        text_h, text_w, _ = text_img.shape

        # create empty text block
        cv2.rectangle(text_img, (0, 0), (text_w, text_h), (0, 0, 0), -1)

        # text
        text = f"Frame index = {frame_idx}"
        cv2.putText(text_img, text, (int(text_w * 0.01), int(text_h * 0.9)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)


    def main(self, frame_idx, pred_pose, gt_pose, ref_2d, cur_2d, ref_img, cur_img, inlier=None, N=127):

        # Trajectory visualization
        # if self.cfg.trajectory.vis_traj:
        #     self.draw_traj(frame_idx, pred_pose, gt_pose)

        

        # self.draw_frame(cur_img)

        # # temporal match
        # if vo.cfg.visualization.kp_match.vis_temp.enable and \
        #      vo.tracking_stage >= 1:
        #         self.draw_match_temp(vo=vo)
        

        # side-by-side match
        if self.cfg.kp_match.vis_side.enable:
            self.draw_match_side(ref_2d, cur_2d, ref_img, cur_img, inlier, N)


        self.draw_text(frame_idx)

        # # Depth
        # if vo.cfg.visualization.depth.depth_disp is not None:
        #     self.draw_depth(vo)





        # # Forward Flow
        # if vo.cfg.visualization.flow.vis_forward_flow and \
        #     vo.tracking_stage >= 1 and \
        #         vo.ref_data.get('flow') is not None:
        #             self.draw_flow(vo.ref_data['flow'], 'flow1')
        
        # # Backward Flow
        # if vo.cfg.visualization.flow.vis_backward_flow and \
        #     vo.tracking_stage >= 1 and \
        #         vo.cur_data.get('flow') is not None:
        #             self.draw_flow(vo.cur_data['flow'], 'flow2')
        
        # # Forward-backward flow consistency
        # if vo.cfg.visualization.flow.vis_flow_diff and \
        #     vo.cfg.deep_flow.forward_backward and \
        #       vo.tracking_stage >= 1:
        #         self.draw_flow_consistency(vo)
        
        # # Optical-Rigid flow consistency
        # if vo.cfg.visualization.flow.vis_rigid_diff and \
        #       vo.tracking_stage >= 1:
        #         self.draw_rigid_flow_consistency(vo)
        
        # # FIXME: (Experiment Ver. only) draw warp_diff
        # # if vo.tracking_stage >= 1:
        # #     self.draw_warp_diff(vo)

        # # Save visualization result
        # if vo.cfg.visualization.save_img:
        #     img_dir_path = os.path.join(
        #         vo.cfg.directory.result_dir, "img_{}".format(vo.cfg.seq))
        #     mkdir_if_not_exists(img_dir_path)
        #     img_path = os.path.join(img_dir_path, "{:06d}.jpg".format(vo.cur_data['id']))
        #     cv2.imwrite(img_path, self.img)
        
        cv2.imshow('DPDM', self.img)
        cv2.waitKey(1)

        self.interface()


if __name__ == '__main__':
    fd = FrameDrawer() 