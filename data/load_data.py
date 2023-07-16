import re
import os
import sys
import copy
import json
import glob
import random
import pickle
import warnings
import torch
import torchvision
from pathlib import Path
import numpy as np
from numpy import int32  # , result_type
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from scipy.interpolate import UnivariateSpline

from data.tools import *

sys.path.append('./')
warnings.simplefilter('ignore', np.RankWarning)



def homographic_transformation(Matrix, x, y):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x3 homography matrix
            x (array): original x coordinates
            y (array): original y coordinates
    """
    ones = np.ones((1, len(y)))
    coordinates = np.vstack((x, y, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :] / trans[2, :]
    y_vals = trans[1, :] / trans[2, :]
    return x_vals, y_vals


def prune_3d_lane_by_visibility(lane_3d, visibility):
    lane_3d = lane_3d[visibility > 0, ...]
    return lane_3d

#我们不使用相机投影模型，而是使用单应性度量将道路平面上的 3d 点投影到图像平面。 在这种假设下，尺寸 2（道路的高度）将被省略。
def homograpthy_g2im_extrinsic(E, K):
    """E: extrinsic matrix, 4*4"""
    E_inv = np.linalg.inv(E)[0:3, :]
    H_g2c = E_inv[:, [0, 1, 3]]
    H_g2im = np.matmul(K, H_g2c)
    return H_g2im




def prune_3d_lane_by_range(lane_3d, x_min, x_max):
    # TODO: solve hard coded range later
    # remove points with y out of range
    # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
    # 2D dataset requires this to rule out those points projected to ground, but out of meaningful range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 1] > 0, lane_3d[:, 1] < 200), ...]

    # remove lane points out of x range
    lane_3d = lane_3d[np.logical_and(lane_3d[:, 0] > x_min,
                                     lane_3d[:, 0] < x_max), ...]
    return lane_3d


def projective_transformation(Matrix, x, y, z):
    """
    Helper function to transform coordinates defined by transformation matrix

    Args:
            Matrix (multi dim - array): 3x4 projection matrix
            x (array): original x coordinates
            y (array): original y coordinates
            z (array): original z coordinates
    """
    ones = np.ones((1, len(z)))
    coordinates = np.vstack((x, y, z, ones))
    trans = np.matmul(Matrix, coordinates)

    x_vals = trans[0, :] / trans[2, :]
    y_vals = trans[1, :] / trans[2, :]
    return x_vals, y_vals


def convert_lanes_3d_to_gflat(lanes, P_g2gflat):
    """
        Convert a set of lanes from 3D ground coordinates [X, Y, Z], to IPM-based
        flat ground coordinates [x_gflat, y_gflat, Z]
    :param lanes: a list of N x 3 numpy arrays recording a set of 3d lanes
    :param P_g2gflat: projection matrix from 3D ground coordinates to flat ground coordinates
    :return:
    """
    # TODO: this function can be simplified with the derived formula
    for lane in lanes:
        # convert gt label to anchor label
        lane_gflat_x, lane_gflat_y = projective_transformation(P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2])
        lane[:, 0] = lane_gflat_x
        lane[:, 1] = lane_gflat_y


def make_lane_y_mono_inc(lane):
    """
        Due to lose of height dim, projected lanes to flat ground plane may not have monotonically increasing y.
        This function trace the y with monotonically increasing y, and output a pruned lane
    :param lane:
    :return:
    """
    idx2del = []
    max_y = lane[0, 1]
    for i in range(1, lane.shape[0]):
        # hard-coded a smallest step, so the far-away near horizontal tail can be pruned
        if lane[i, 1] <= max_y + 3:
            idx2del.append(i)
        else:
            max_y = lane[i, 1]
    lane = np.delete(lane, idx2del, 0)
    return lane



def data_aug_rotate(img):
    # assume img in PIL image format
    rot = random.uniform(-np.pi / 18, np.pi / 18)
    # rot = random.uniform(-10, 10)
    center_x = img.width / 2
    center_y = img.height / 2
    rot_mat = cv2.getRotationMatrix2D((center_x, center_y), rot, 1.0)
    img_rot = np.array(img)
    img_rot = cv2.warpAffine(img_rot, rot_mat, (img.width, img.height), flags=cv2.INTER_LINEAR)
    # img_rot = img.rotate(rot)
    # rot = rot / 180 * np.pi
    rot_mat = np.vstack([rot_mat, [0, 0, 1]])
    return img_rot, rot_mat


def homography_crop_resize(org_img_size, crop_y, resize_img_size):
    """
        compute the homography matrix transform original image to cropped and resized image
    :param org_img_size: [org_h, org_w]
    :param crop_y:
    :param resize_img_size: [resize_h, resize_w]
    :return:
    """
    # transform original image region to network input region
    ratio_x = resize_img_size[1] / org_img_size[1]
    ratio_y = resize_img_size[0] / (org_img_size[0] - crop_y)
    H_c = np.array([[ratio_x, 0, 0],
                    [0, ratio_y, -ratio_y * crop_y],
                    [0, 0, 1]])
    return H_c


class lane_dataset(Dataset):
    # dataset_base_dir is image path, json_file_path is json file path,
    def __init__(self, dataset_base_dir, json_file_path, extend_dataset_base_dir,
     extend_json_file_path, args, data_aug_conf, is_data_aug=False, save_std=False):

        # define image pre-processor
        self.totensor = transforms.ToTensor()
        # expect same mean/std for all torchvision models
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        #self.normalize = transforms.Normalize(args.vgg_mean, args.vgg_std)

        self.normalize = torchvision.transforms.Compose((
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(mean=args.vgg_mean,
                                        std=args.vgg_std),
        ))

        self.is_data_aug = is_data_aug
        self.data_aug_conf = data_aug_conf
         
        self.dataset_base_dir = dataset_base_dir
        self.json_file_path = json_file_path
        self.extend_dataset_base_dir = extend_dataset_base_dir
        self.extend_json_file_path = extend_json_file_path

        self.num_category = args.num_category

        # dataset parameters
        self.camera_nums = args.camera_nums
        # self.h_org = args.org_h
        # self.w_org = args.org_w
        # self.h_crop = args.crop_y

        #计算ahchor 的参数
        # parameters related to service network
        # self.h_net = args.resize_h
        # self.w_net = args.resize_w
        # self.ipm_h = args.ipm_h
        self.ipm_w = args.ipm_w
        self.top_view_region = args.top_view_region


        # TODO: need it?
        # transformation from ipm to ground region
        self.max_lanes = args.max_lanes

        self.fix_cam = False

        self.cam_R = []
        #camera 0
        self.cam_R.append(np.array([[0, 0, 1],
                               [-1, 0, 0],
                               [0, -1, 0]], dtype=float))
        #camera 1
        self.cam_R.append(np.linalg.inv(np.array([[0, -np.cos(np.pi / 4), np.sin(np.pi / 4)  ],
                                             [0, -np.sin(np.pi / 4), -np.cos(np.pi / 4) ],
                                             [1, 0                 , 0                  ]], dtype=float)))

        #camera 2
        self.cam_R.append(np.linalg.inv(np.array([[-1, 0               , 0                  ],
                                             [0, np.sin(np.pi / 4), np.cos(np.pi / 4)  ],
                                             [0, np.cos(np.pi / 4), -np.sin(np.pi / 4) ]], dtype=float)))
        
        #camera 3
        self.cam_R.append(np.linalg.inv(np.array([[-1, 0, 0],
                                             [0, 0, -1],
                                             [0, -1, 0]], dtype=float)))

        #camera 4
        self.cam_R.append(np.linalg.inv(np.array([[-1, 0, 0],
                                             [0,  0, 1],
                                             [0,  1, 0]], dtype=float)))



        # ======= step 1 =======
        # compute anchor steps
        self.x_min, self.x_max = self.top_view_region[0, 0], self.top_view_region[1, 0]
        self.y_min, self.y_max = self.top_view_region[2, 1], self.top_view_region[0, 1]
        self.anchor_num_before_shear = self.ipm_w // 8
        self.anchor_x_steps = np.linspace(self.x_min, self.x_max, self.anchor_num_before_shear, endpoint=True)
        self.anchor_y_steps = args.anchor_y_steps
        self.num_y_steps = len(self.anchor_y_steps)

        # compute anchor grid with different far center points
        # currently, anchor grid consists of [center, left-sheared, right-sheared] concatenated
        self.anchor_num = self.anchor_num_before_shear * 7
        self.anchor_grid_x = np.repeat(np.expand_dims(self.anchor_x_steps, axis=1), self.num_y_steps,
                                       axis=1)  # center
        anchor_grid_y = np.repeat(np.expand_dims(self.anchor_y_steps, axis=0), self.anchor_num_before_shear, axis=0)

        x2y_ratio = self.x_min / (self.y_max - self.y_min)  # x change per unit y change (for left-sheared anchors)
        anchor_grid_x_left_10 = (anchor_grid_y - self.y_min) * x2y_ratio + self.anchor_grid_x
        # right-sheared anchors are symmetrical to left-sheared ones
        anchor_grid_x_right_10 = np.flip(-anchor_grid_x_left_10, axis=0)
        x2y_ratio = (self.x_min - self.x_max) / (
                self.y_max - self.y_min)  # x change per unit y change (for left-sheared anchors)
        anchor_grid_x_left_20 = (anchor_grid_y - self.y_min) * x2y_ratio + self.anchor_grid_x
        # right-sheared anchors are symmetrical to left-sheared ones
        anchor_grid_x_right_20 = np.flip(-anchor_grid_x_left_20, axis=0)
        x2y_ratio = 2.0 * (self.x_min - self.x_max) / (
                self.y_max - self.y_min)  # x change per unit y change (for left-sheared anchors)
        anchor_grid_x_left_40 = (anchor_grid_y - self.y_min) * x2y_ratio + self.anchor_grid_x
        # right-sheared anchors are symmetrical to left-sheared ones
        anchor_grid_x_right_40 = np.flip(-anchor_grid_x_left_40, axis=0)
        # concat the three parts
        self.anchor_grid_x = np.concatenate((self.anchor_grid_x,
                                             anchor_grid_x_left_10, anchor_grid_x_right_10,
                                             anchor_grid_x_left_20, anchor_grid_x_right_20,
                                             anchor_grid_x_left_40, anchor_grid_x_right_40), axis=0)
        args.anchor_grid_x = self.anchor_grid_x

        self.num_types = 1
        self.anchor_dim = 3 * self.num_y_steps + args.num_category
        self.y_ref = args.y_ref
        self.ref_id = np.argmin(np.abs(self.num_y_steps - self.y_ref))

        self.save_json_path = args.save_json_path

        # ======= step 2 =======
        # parse ground-truth file
        self._x_off_std, \
        self._z_std  = self.init_dataset_openlane(dataset_base_dir, json_file_path, 
            extend_dataset_base_dir, extend_json_file_path)

        # Note: samples
        self.n_samples = len(self._label_list)

        if save_std is True:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            with open(os.path.join(args.save_path, 'geo_anchor_std.json'), 'w') as jsonFile:
                json_out = {}
                json_out["x_off_std"] = self._x_off_std.tolist()
                json_out["z_std"] = self._z_std.tolist()
                json.dump(json_out, jsonFile)
                jsonFile.write('\n')

    def __len__(self):
        return self.n_samples

    # new getitem, WIP
    def WIP__getitem__(self, idx):

        idx_json_file = self._label_list[idx]

        # ======= step 1 =======
        # preprocess data from json file
        img_name, front_extrinsic, extrinsics, intrinsics, \
        gt_lanes, gt_vis_inds, gt_category_3d, \
        _laneline_ass_id = self.preprocess_data_from_json_openlane(idx_json_file)

        image, images, all_extrinsics, all_intrinsics, all_rots, all_trans, all_post_rots, all_post_trans=self.image_data_get(
            idx_json_file, img_name, front_extrinsic, intrinsics, extrinsics)


        gt_anchor = np.zeros([self.anchor_num, self.num_types, self.anchor_dim], dtype=np.float32)
        gt_anchor[:, :, self.anchor_dim - self.num_category] = 1.0
        # gt_laneline_img = self._gt_laneline_im_all[idx]
        for i in range(len(gt_lanes)):
            # if ass_id >= 0:
            ass_id = _laneline_ass_id[i]

            x_off_values = gt_lanes[i][:, 0]
            z_values = gt_lanes[i][:, 1]
            visibility = gt_vis_inds[i]
            # assign anchor tensor values
            gt_anchor[ass_id, 0, 0: self.num_y_steps] = x_off_values
            gt_anchor[ass_id, 0, self.num_y_steps:2 * self.num_y_steps] = z_values
            gt_anchor[ass_id, 0, 2 * self.num_y_steps:3 * self.num_y_steps] = visibility

            # gt_anchor[ass_id, 0, -1] = 1.0

            gt_anchor[ass_id, 0, self.anchor_dim - self.num_category] = 0.0
            gt_anchor[ass_id, 0, self.anchor_dim - self.num_category + gt_category_3d[i]] = 1.0

        # if self.is_data_aug:
        #     img_rot, aug_mat = data_aug_rotate(image)
        #     image = Image.fromarray(img_rot)
        # image = self.totensor(image).float()
        image = self.normalize(image)
        gt_anchor = gt_anchor.reshape([self.anchor_num, -1])
        gt_anchor = torch.from_numpy(gt_anchor)
        intrinsics = torch.from_numpy(intrinsics)
        extrinsics = torch.from_numpy(extrinsics)


        trans = extrinsics[0:3, 3]
        rots = extrinsics[0:3, 0:3]
        #TODO, change image shape, add camera nums = 1
        image = torch.unsqueeze(image, 0)
        intrinsics = torch.unsqueeze(intrinsics, 0)
        extrinsics = torch.unsqueeze(extrinsics, 0)
        trans = torch.unsqueeze(trans, 0)
        rots = torch.unsqueeze(rots, 0)

        if self.is_data_aug:
            # aug_mat = torch.from_numpy(aug_mat.astype(np.float32))
            aug_mat=0
            return idx_json_file, image, gt_anchor, idx, intrinsics, extrinsics, aug_mat, rots,trans, images, all_intrinsics, all_extrinsics,all_rots, all_trans,all_post_rots, all_post_trans
        return idx_json_file, image, gt_anchor, idx, intrinsics, extrinsics, rots,trans, images, all_intrinsics, all_extrinsics,all_rots, all_trans, all_post_rots, all_post_trans

    # old getitem, workable
    def __getitem__(self, idx):
        return self.WIP__getitem__(idx)

    def preprocess_data_from_json_openlane(self, idx_json_file):

        _label_image_path = None
        front_extrinsic=None
        cam_extrinsics = None
        cam_intrinsics = None



        # print(idx_json_file)
        with open(idx_json_file, 'r') as file:
            file_lines = [line for line in file]
            info_dict = json.loads(file_lines[0])  # 因为只有1行

            image_path = os.path.join(self.dataset_base_dir, info_dict['file_path'])
            assert os.path.exists(image_path), '{:s} not exist'.format(image_path)
            #image  的地址
            _label_image_path = image_path

            front_extrinsic = np.array(info_dict['extrinsic'])
            # Re-calculate extrinsic matrix based on ground coordinate
            cam_extrinsics = self.extrinsic_recalculate(front_extrinsic, front_extrinsic, 0)

            cam_intrinsics = info_dict['intrinsic']
            cam_intrinsics = np.array(cam_intrinsics)

            gt_lanes_packed = info_dict['lane_lines']
            gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
            for i, gt_lane_packed in enumerate(gt_lanes_packed):
                # A GT lane can be either 2D or 3D
                # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                lane = np.array(gt_lane_packed['xyz'])
                lane_visibility = np.array(gt_lane_packed['visibility'])

                # Coordinate convertion for openlane_300 data
                lane = np.vstack((lane, np.ones((1, lane.shape[1]))))
                cam_representation = np.linalg.inv(
                    np.array([[0, 0, 1, 0],
                              [-1, 0, 0, 0],
                              [0, -1, 0, 0],
                              [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
                lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))

                lane = lane[0:3, :].T
                gt_lane_pts.append(lane)
                gt_lane_visibility.append(lane_visibility)

                #合并左侧和右侧的 为 路的边缘， 21 -> 20
                if 'category' in gt_lane_packed:
                    lane_cate = gt_lane_packed['category']
                    if lane_cate == 21:  # merge left and right road edge into road edge
                        lane_cate = 20
                    gt_laneline_category.append(lane_cate)
                else:
                    gt_laneline_category.append(1)

        # _label_laneline_org = copy.deepcopy(gt_lane_pts)
        # _gt_laneline_category_org = copy.deepcopy(np.array(gt_laneline_category))

        cam_K = cam_intrinsics
        cam_E = cam_extrinsics
        P_g2im = projection_g2im_extrinsic(cam_E, cam_K) # ground truth Project to top view Matrix 
        H_g2im = homograpthy_g2im_extrinsic(cam_E, cam_K) # project the 3d point on the road plane to the image plane
        H_im2g = np.linalg.inv(H_g2im)
        
        # 通常情况下，地面坐标系是一个三维坐标系，用于描述物体在三维空间中的位置。
        # 然而，在某些应用中，我们可能更关注物体在地面平面上的位置，例如道路上的车道线或车辆的位置。
        # 因此，我们需要将物体的三维坐标转换到地面平面上的二维坐标。
        # 变换矩阵 P_g2gflat 是由代码中的投影矩阵 P_g2im 和  从图像坐标系到地面坐标系的单应矩阵 H_im2g 相乘得到的。

        # 这个变换矩阵可以将从地面坐标系表示的点或物体的坐标投影到地面平面上，得到地面平面上的坐标。
        
        # 换句话说，通过应用变换矩阵 P_g2gflat，我们可以将车道线或其他物体在地面坐标系中的三维坐标转换为在地面平面上的二维坐标。
        # 这可以方便地对地面上的物体进行分析、测量或其他操作，而不需要考虑物体在垂直方向上的位置。


        P_g2gflat = np.matmul(H_im2g, P_g2im)

        gt_lanes = gt_lane_pts
        gt_visibility = gt_lane_visibility
        gt_category = gt_laneline_category


        #过滤车道线上的点（过滤不可见点， 离群点）
        # prune gt lanes by visibility labels
        gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]) for k, gt_lane in enumerate(gt_lanes)]
        # _label_laneline_org = copy.deepcopy(gt_lanes)


        #TODO:离群点重新定义。。。 为啥 0 < y < 200? 
        # prune out-of-range points are necessary before transformation
        #在群点的定义： 1： 0< y <200     ， 2： 3 * x_min(-10) < x < 3*x_max(10) ， 3： 车道线的数量大于1个
        # 3D label may miss super long straight-line with only two points: Not have to be 200, gt need a min-step
        gt_lanes = [prune_3d_lane_by_range(gt_lane, 3 * self.x_min, 3 * self.x_max) for gt_lane in gt_lanes]
        gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

#  将一组车道从 3D 地面坐标 [X, Y, Z] 转换为基于 IPM 的平坦地面坐标 [x_gflat, y_gflat, Z]。 
#  具体的转换过程：首先将x,y ， z通过投影矩阵算出来，之后x，y都除以z，得到新得x, y 。。。。（这个操作是假设z = 1, 可以查看投影公式）， 得到得x, y替换lane得值。
        # convert 3d lanes to flat ground space
        self.convert_lanes_3d_to_gflat(gt_lanes, P_g2gflat)

        gt_anchors = []
        ass_ids = []
        visibility_vectors = []
        category_ids = []

        for i in range(len(gt_lanes)):

            #ass_id：anchor 所属的id， gt_anchors：lane的anchor label ，  
            # convert gt label to anchor label
            # consider individual out-of-range interpolation still visible
            ass_id, x_off_values, z_values, visibility_vec = self.convert_label_to_anchor(gt_lanes[i])
            if ass_id >= 0:
                gt_anchors.append(np.vstack([x_off_values, z_values]).T)
                ass_ids.append(ass_id)
                visibility_vectors.append(visibility_vec)
                category_ids.append(gt_category[i])

        # normalize x anad z, in replacement of normalize_lane_label
        for lane in gt_anchors:
            lane[:, 0] = np.divide(lane[:, 0], self._x_off_std)
            lane[:, 1] = np.divide(lane[:, 1], self._z_std)

        return _label_image_path, front_extrinsic, cam_extrinsics, cam_intrinsics, \
               gt_anchors, visibility_vectors, category_ids, ass_ids


    def get_extend_file(self, dst_path, num, ref_file):
        (orig_path, orig_file) = os.path.split(ref_file)
        (_, tf_tag) = os.path.split(orig_path)
        (filename,extension) = os.path.splitext(orig_file)
        #construct extend name
        dst_file = dst_path  + tf_tag + '/' + filename + '.' + str(num) + extension
        return dst_file

    # Re-calculate extrinsic matrix based on ground coordinate
    def extrinsic_recalculate(self, extrinsic, front_extrinsic, cam_num):
        R_vg = np.array([[0, 1, 0],
                            [-1, 0, 0],
                            [0, 0, 1]], dtype=float)
        # R_gc = np.array([[1, 0, 0],
        #                     [0, 0, 1],
        #                     [0, -1, 0]], dtype=float)

        extrinsic[:3, :3] = np.matmul(
                                    np.matmul(np.linalg.inv(R_vg), extrinsic[:3, :3]),
                                        self.cam_R[cam_num])
        
        # extrinsic[:3, :3] = np.matmul(np.matmul(
        #     np.matmul(np.linalg.inv(R_vg), extrinsic[:3, :3]),
        #     R_vg), R_gc)


        #Translate  camera coordinate to 3D-LanNet vehicle coordinate
        if cam_num == 0:
            extrinsic[0:2, 3] = 0.0
        else:
            extrinsic[0:3, 3] = extrinsic[0:3, 3] - front_extrinsic[0:3, 3]

        return extrinsic

    def json_file_path_save(self, label_image_path, json_file_path):

        idx_path = {}
        for i, path in enumerate(label_image_path):
            idx_path[i] = path

        if "lane3d_1000" in json_file_path:
            train_idx_file = os.path.join(self.save_json_path, 'train_idx_1000.json')
            val_idx_file = os.path.join(self.save_json_path, 'val_idx_1000.json')
        elif "lane3d_300" in json_file_path:
            train_idx_file = os.path.join(self.save_json_path, 'train_idx_300.json')
            val_idx_file = os.path.join(self.save_json_path, 'val_idx_300.json')
        else:
            raise Exception("openlane version not supported")

        if not os.path.exists(self.save_json_path):
            os.makedirs(self.save_json_path)
        if not os.path.isfile(train_idx_file):
            with open(train_idx_file, 'w') as f:
                json.dump(idx_path, f)
        elif not os.path.isfile(val_idx_file):
            with open(val_idx_file, 'w') as f:
                json.dump(idx_path, f)


    # convert labeled laneline to anchor format
    def label_lane_convert(self, cam_intrinsics_all, cam_extrinsics_all,
        gt_laneline_visibility_all, gt_laneline_category_all, gt_laneline_pts_all_org, gt_laneline_pts_all):

        gt_laneline_ass_ids = []
        lane_x_off_all = []
        lane_z_all = []
        lane_y_off_all = []  # this is the offset of y when transformed back 3 3D
        visibility_all_flat = []
        #gt_laneline_im_all = []
        for idx in range(len(gt_laneline_pts_all)):
            cam_K = cam_intrinsics_all[idx]
            cam_E = cam_extrinsics_all[idx]
            P_g2im = projection_g2im_extrinsic(cam_E, cam_K)
            H_g2im = homograpthy_g2im_extrinsic(cam_E, cam_K)

            H_im2g = np.linalg.inv(H_g2im)

            P_g2gflat = np.matmul(H_im2g, P_g2im)

            gt_lanes = gt_laneline_pts_all[idx]
            gt_visibility = gt_laneline_visibility_all[idx]
            gt_category = gt_laneline_category_all[idx]

            # prune gt lanes by visibility labels
            gt_lanes = [prune_3d_lane_by_visibility(gt_lane, gt_visibility[k]) for k, gt_lane in enumerate(gt_lanes)]
            gt_laneline_pts_all_org[idx] = gt_lanes

            # project gt laneline to image plane
            # gt_laneline_im = []
            # for gt_lane in gt_lanes:
            #     x_vals, y_vals = projective_transformation(P_g2im, gt_lane[:, 0], gt_lane[:, 1], gt_lane[:, 2])
            #     gt_laneline_im_oneline = np.array([x_vals, y_vals]).T.tolist()
            #     gt_laneline_im.append(gt_laneline_im_oneline)
            # gt_laneline_im_all.append(gt_laneline_im)

            # prune out-of-range points are necessary before transformation
            gt_lanes = [prune_3d_lane_by_range(gt_lane, 3 * self.x_min, 3 * self.x_max) for gt_lane in gt_lanes]
            gt_lanes = [lane for lane in gt_lanes if lane.shape[0] > 1]

            # convert 3d lanes to flat ground space
            self.convert_lanes_3d_to_gflat(gt_lanes, P_g2gflat)

            gt_anchors = []
            ass_ids = []
            visibility_vectors = []
            category_ids = []

            self.new_match = False
            # if self.new_match:
            #     frame_x_off_values, frame_z_values, frame_visibility_vectors = [], [], []
            for i in range(len(gt_lanes)):

                # convert gt label to anchor label
                # consider individual out-of-range interpolation still visibleconvert_label_to_anchor
                ass_id, x_off_values, z_values, visibility_vec = self.convert_label_to_anchor(gt_lanes[i])

                if ass_id >= 0:
                    gt_anchors.append(np.vstack([x_off_values, z_values]).T)
                    ass_ids.append(ass_id)
                    visibility_vectors.append(visibility_vec)
                    category_ids.append(gt_category[i])

            for i in range(len(gt_anchors)):
                lane_x_off_all.append(gt_anchors[i][:, 0])
                lane_z_all.append(gt_anchors[i][:, 1])
                # compute y offset when transformed back to 3D space
            visibility_all_flat.extend(visibility_vectors)
            gt_laneline_ass_ids.append(ass_ids)
            gt_laneline_pts_all[idx] = gt_anchors
            gt_laneline_visibility_all[idx] = visibility_vectors
            gt_laneline_category_all[idx] = category_ids

        lane_x_off_all = np.array(lane_x_off_all)
        lane_z_all = np.array(lane_z_all)
        visibility_all_flat = np.array(visibility_all_flat)

        # computed weighted std based on visibility
        lane_x_off_std = np.sqrt(np.average(lane_x_off_all ** 2, weights=visibility_all_flat, axis=0))
        lane_z_std = np.sqrt(np.average(lane_z_all ** 2, weights=visibility_all_flat, axis=0))

        return lane_x_off_std, lane_z_std


    def sample_augmentation(self,cam_num):

        if cam_num < 3:
            H, W = self.data_aug_conf['FRONT_H'], self.data_aug_conf['FRONT_W']
        else:
            H, W = self.data_aug_conf['SIDE_H'], self.data_aug_conf['SIDE_W']

        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate



    def image_data_get(self, label_file, main_image_path,front_extrinsic, main_instrinsic, main_extrinsic):

        images = []
        all_intrinsics = []
        all_extrinsics = []

        all_trans = []
        all_rots = []

        all_post_rots = []
        all_post_trans = []

        #base
        main_image=''
        with open(main_image_path, 'rb') as f:
            main_image = (Image.open(f).convert('RGB'))

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(0)
            main_image, post_rot2, post_tran2 = img_transform(main_image, post_rot, post_tran,
                        resize, resize_dims, crop,
                        flip, rotate)
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            all_post_rots.append(post_rot)
            all_post_trans.append(post_tran)


            # #数据增强可以在这里完成
            # # image preprocess with crop and resize
            # main_image = F.crop(main_image, self.h_crop, 0, self.h_org - self.h_crop, self.w_org)
            # main_image = F.resize(main_image, size=(self.h_net, self.w_net), interpolation=InterpolationMode.BILINEAR)
           
           
            images.append(self.normalize(main_image)) #注意这里图像数据没有转换成tensor


        main_instrinsic=torch.Tensor(main_instrinsic)
        main_extrinsic= torch.Tensor(main_extrinsic)
        all_intrinsics.append(main_instrinsic)
        all_extrinsics.append(main_extrinsic)

        trans = main_extrinsic[0:3, 3]
        rots = main_extrinsic[0:3, 0:3]
        all_trans.append(trans)
        all_rots.append(rots)

        # print("camera numbers: {:d}".format(self.camera_nums))

        for extend_num in range(1, self.camera_nums):
            #extend label 
            extend_label_file = self.get_extend_file(self.extend_json_file_path ,extend_num, label_file)

            #extennd image
            dataset_class='training/'
            if "validation/" in main_image_path:
                dataset_class='validation/'
            extend_image_path =  self.get_extend_file(self.extend_dataset_base_dir + dataset_class,
                extend_num, main_image_path)
                        
            # print(extend_image_path)
            with open(extend_label_file + '', 'r') as file:
                info_dict = json.load(file)
                intrinsic = np.array(info_dict['intrinsic'])
                extrinsic = np.array(info_dict['extrinsic'])
                # Re-calculate extrinsic matrix based on ground coordinate
                extrinsic = self.extrinsic_recalculate(extrinsic,front_extrinsic, extend_num)

                intrinsic=torch.Tensor(intrinsic)
                extrinsic=torch.Tensor(extrinsic)
                all_intrinsics.append(intrinsic)
                all_extrinsics.append(extrinsic)  
                #add by wucunyin.....
                trans = extrinsic[0:3, 3]
                rots = extrinsic[0:3, 0:3]
                all_trans.append(trans)
                all_rots.append(rots)

                with open(extend_image_path, 'rb') as f:
                    img = (Image.open(f).convert('RGB'))

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation(extend_num)
                    img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                resize, resize_dims, crop,
                                flip, rotate)
                    
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    all_post_rots.append(post_rot)
                    all_post_trans.append(post_tran)

                    # #数据增强可以在这里完成
                    # # image preprocess with crop and resize
                    # img = F.crop(img, self.h_crop, 0, self.h_org - self.h_crop, self.w_org)
                    # img = F.resize(img, size=(self.h_net, self.w_net), interpolation=InterpolationMode.BILINEAR)

                    images.append(self.normalize(img)) #注意这里图像数据没有转换成tensor
    
        return (main_image, torch.stack(images), torch.stack(all_extrinsics), 
                torch.stack(all_intrinsics), torch.stack(all_rots), torch.stack(all_trans) , 
                torch.stack(all_post_rots), torch.stack(all_post_trans))





    def init_dataset_openlane(self, dataset_base_dir, json_file_path, extend_dataset_base_dir, extend_json_file_path):

        #所有的前相机的json文件
        label_list = glob.glob(json_file_path + '**/*.json', recursive=True)
        # save label list and this determine the idx order
        self._label_list = label_list


        #缓存用于加速
        # accelerate openlane dataset IO
        Path("./.cache/").mkdir(parents=True, exist_ok=True)

        if "training/" in json_file_path:
            self.is_train = True
            if "lane3d_1000" in json_file_path:
                if os.path.isfile("./.cache/openlane_1000_preprocess_train_newanchor.pkl"):
                    with open("./.cache/openlane_1000_preprocess_train_newanchor.pkl", "rb") as f:
                        cache_file = pickle.load(f)
                        return self.read_cache_file_beta(cache_file)
            else:
                if os.path.isfile("./.cache/openlane_preprocess_train_newanchor.pkl"):
                    # TODO: need to change later
                    with open("./.cache/openlane_preprocess_train_newanchor.pkl", "rb") as f:
                        cache_file = pickle.load(f)
                        return self.read_cache_file_beta(cache_file)

        elif "validation/" in json_file_path:
            self.is_train = False
            if "lane3d_1000" in json_file_path:
                if os.path.isfile("./.cache/openlane_1000_preprocess_valid_newanchor.pkl"):
                    with open("./.cache/openlane_1000_preprocess_valid_newanchor.pkl", "rb") as f:
                        cache_file = pickle.load(f)
                        return self.read_cache_file_beta(cache_file)
            else:
                if os.path.isfile("./.cache/openlane_preprocess_valid_newanchor.pkl"):
                    # TODO: need to change later
                    with open("./.cache/openlane_preprocess_valid_newanchor.pkl", "rb") as f:
                        cache_file = pickle.load(f)
                        return self.read_cache_file_beta(cache_file)



        elif "test/" in json_file_path:
            self.is_train = False
            if "lane3d_1000" in json_file_path:
                if os.path.isfile("./.cache/openlane_1000_preprocess_test_newanchor.pkl"):
                    with open("./.cache/openlane_1000_preprocess_test_newanchor.pkl", "rb") as f:
                        cache_file = pickle.load(f)
                        return self.read_cache_file_beta(cache_file)
            else:
                if os.path.isfile("./.cache/openlane_preprocess_test_newanchor.pkl"):
                    # TODO: need to change later
                    with open("./.cache/openlane_preprocess_test_newanchor.pkl", "rb") as f:
                        cache_file = pickle.load(f)
                        return self.read_cache_file_beta(cache_file)



        # load image path, and lane pts
        label_image_path = []
        gt_laneline_pts_all = []
        gt_laneline_visibility_all = []
        gt_laneline_category_all = []
        cam_intrinsics_all = []
        cam_extrinsics_all = []

        for label_file in label_list:
            image_path = ''

            with open(label_file, 'r') as file:
                file_lines = [line for line in file]
                info_dict = json.loads(file_lines[0])

                image_path = os.path.join(dataset_base_dir, info_dict['file_path'])
                assert os.path.exists(image_path), '{:s} not exist'.format(image_path)

                label_image_path.append(image_path)

                # ======= step 3 =======
                # get  extrinsic and intrinsic
                cam_extrinsics = np.array(info_dict['extrinsic'])
                ########################remove?###############################
                cam_extrinsics = self.extrinsic_recalculate(cam_extrinsics, cam_extrinsics, 0)
                cam_extrinsics_all.append(cam_extrinsics)

                # gt_cam_height = cam_extrinsics[2, 3]
                # gt_cam_height_all.append(gt_cam_height)

                #########################done##################################
                cam_intrinsics = info_dict['intrinsic']
                cam_intrinsics = np.array(cam_intrinsics)
                cam_intrinsics_all.append(cam_intrinsics)


                # ======= step 4 =======
                # get lane pts, visibility, category.
                gt_lanes_packed = info_dict['lane_lines']
                gt_lane_pts, gt_lane_visibility, gt_laneline_category = [], [], []
                for i, gt_lane_packed in enumerate(gt_lanes_packed):
                    # A GT lane can be either 2D or 3D
                    # if a GT lane is 3D, the height is intact from 3D GT, so keep it intact here too
                    lane = np.array(gt_lane_packed['xyz'])
                    lane_visibility = np.array(gt_lane_packed['visibility'])

                    # Coordinate convertion for openlane_300 data
                    lane = np.vstack((lane, np.ones((1, lane.shape[1]))))

                    cam_representation = np.linalg.inv(
                        np.array([[0, 0, 1, 0],
                                  [-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 1]], dtype=float))  # transformation from apollo camera to openlane camera
                    lane = np.matmul(cam_extrinsics, np.matmul(cam_representation, lane))

                    lane = lane[0:3, :].T
                    gt_lane_pts.append(lane)
                    gt_lane_visibility.append(lane_visibility)

                    if 'category' in gt_lane_packed:
                        lane_cate = gt_lane_packed['category']
                        if lane_cate == 21:  # merge left and right road edge into road edge
                            lane_cate = 20
                        gt_laneline_category.append(lane_cate)
                    else:
                        gt_laneline_category.append(1)

                    # remove lane too far
                    # if np.amin(gt_lane_pts[i][:, 1]) > 10.0:
                    #     gt_lane_visibility[i] = np.zeros_like(gt_lane_visibility[i])
                gt_laneline_pts_all.append(gt_lane_pts)
                gt_laneline_visibility_all.append(gt_lane_visibility)
                gt_laneline_category_all.append(np.array(gt_laneline_category, dtype=np.int32))


        # # ======= step 5 =======
        # # save img and label index, just for debug?
        # self.json_file_path_save(label_image_path, json_file_path)

        gt_laneline_pts_all_org = copy.deepcopy(gt_laneline_pts_all)
        cam_intrinsics_all = np.array(cam_intrinsics_all)
        cam_extrinsics_all = np.array(cam_extrinsics_all)

        # ======= step 6 =======
        # # convert labeled laneline to anchor format
        lane_x_off_std,lane_z_std = self.label_lane_convert(cam_intrinsics_all, cam_extrinsics_all,
            gt_laneline_visibility_all, gt_laneline_category_all, gt_laneline_pts_all_org, gt_laneline_pts_all)

        cache_file = {}
        cache_file["lane_x_off_std"] = lane_x_off_std
        cache_file["lane_z_std"] = lane_z_std

        if "training/" in json_file_path:
            if "lane3d_1000" in json_file_path:
                with open("./.cache/openlane_1000_preprocess_train_newanchor.pkl", "wb") as f:
                    pickle.dump(cache_file, f)
            else:
                with open("./.cache/openlane_preprocess_train_newanchor.pkl", "wb") as f:
                    pickle.dump(cache_file, f)
        if "validation/" in json_file_path:
            if "lane3d_1000" in json_file_path:
                with open("./.cache/openlane_1000_preprocess_valid_newanchor.pkl", "wb") as f:
                    pickle.dump(cache_file, f)
            else:
                with open("./.cache/openlane_preprocess_valid_newanchor.pkl", "wb") as f:
                    pickle.dump(cache_file, f)

        return lane_x_off_std, lane_z_std

    def read_cache_file_beta(self, cache_file):

        lane_x_off_std = cache_file["lane_x_off_std"]
        lane_z_std = cache_file["lane_z_std"]
        return lane_x_off_std, lane_z_std


    def set_x_off_std(self, x_off_std):
        self._x_off_std = x_off_std

    def set_z_std(self, z_std):
        self._z_std = z_std

    def normalize_lane_label(self):
        for lanes in self._label_laneline_all:
            for lane in lanes:
                lane[:, 0] = np.divide(lane[:, 0], self._x_off_std)
                lane[:, 1] = np.divide(lane[:, 1], self._z_std)

    def convert_lanes_3d_to_gflat(self, lanes, P_g2gflat):
        """
            Convert a set of lanes from 3D ground coordinates [X, Y, Z], to IPM-based
            flat ground coordinates [x_gflat, y_gflat, Z]
        :param lanes: a list of N x 3 numpy arrays recording a set of 3d lanes
        :param P_g2gflat: projection matrix from 3D ground coordinates to frat ground coordinates
        :return:
        """
        # TODO: this function can be simplified with the derived formula
        for lane in lanes:
            # convert gt label to anchor label
            lane_gflat_x, lane_gflat_y = projective_transformation(P_g2gflat, lane[:, 0], lane[:, 1], lane[:, 2])
            lane[:, 0] = lane_gflat_x
            lane[:, 1] = lane_gflat_y

    def compute_visibility_lanes_gflat(self, lane_anchors, ass_ids):
        """
            Compute the visibility of each anchor point in flat ground space. The reasoning requires all the considering
            lanes globally.
        :param lane_anchors: A list of N x 2 numpy arrays where N equals to number of Y steps in anchor representation
                             x offset and z values are recorded for each lane
               ass_ids: the associated id determine the base x value
        :return:
        """
        if len(lane_anchors) == 0:
            return [], [], []

        vis_inds_lanes = []
        # sort the lane_anchors such that lanes are recorded from left to right
        # sort the lane_anchors based on the x value at the closed anchor
        # do NOT sort the lane_anchors by the order of ass_ids because there could be identical ass_ids

        x_refs = [lane_anchors[i][0, 0] + self.anchor_x_steps[ass_ids[i]] for i in range(len(lane_anchors))]
        sort_idx = np.argsort(x_refs)
        lane_anchors = [lane_anchors[i] for i in sort_idx]
        ass_ids = [ass_ids[i] for i in sort_idx]

        min_x_vec = lane_anchors[0][:, 0] + self.anchor_x_steps[ass_ids[0]]
        max_x_vec = lane_anchors[-1][:, 0] + self.anchor_x_steps[ass_ids[-1]]
        for i, lane in enumerate(lane_anchors):
            vis_inds = np.ones(lane.shape[0])
            for j in range(lane.shape[0]):
                x_value = lane[j, 0] + self.anchor_x_steps[ass_ids[i]]
                if x_value < 3 * self.x_min or x_value > 3 * self.x_max:
                    vis_inds[j:] = 0
                # A point with x < the left most lane's current x is considered invisible
                # A point with x > the right most lane's current x is considered invisible
                if x_value < min_x_vec[j] - 0.01 or x_value > max_x_vec[j] + 0.01:
                    vis_inds[j:] = 0
                    break
                # A point with orientation close enough to horizontal is considered as invisible
                if j > 0:
                    dx = lane[j, 0] - lane[j - 1, 0]
                    dy = self.anchor_y_steps[j] - self.anchor_y_steps[j - 1]
                    if abs(dx / dy) > 10:
                        vis_inds[j:] = 0
                        break
            vis_inds_lanes.append(vis_inds)
        return vis_inds_lanes, lane_anchors, ass_ids

    def convert_label_to_anchor(self, laneline_gt):
        """
            Convert a set of ground-truth lane points to the format of network anchor representation.

            All the given laneline only include visible points. The interpolated points will be marked invisible
        :param laneline_gt: a list of arrays where each array is a set of point coordinates in [x, y, z]
        :return: ass_id: the column id of current lane in anchor representation
                 x_off_values: current lane's x offset from it associated anchor column
                 z_values: current lane's z value in ground coordinates
        """

        gt_lane_3d = laneline_gt

        # prune out points not in valid range, requires additional points to interpolate better
        # prune out-of-range points after transforming to flat ground space, update visibility vector
        valid_indices = np.logical_and(np.logical_and(gt_lane_3d[:, 1] > 0, gt_lane_3d[:, 1] < 200),
                                       np.logical_and(gt_lane_3d[:, 0] > 3 * self.x_min,
                                                      gt_lane_3d[:, 0] < 3 * self.x_max))
        gt_lane_3d = gt_lane_3d[valid_indices, ...]
        # use more restricted range to determine deletion or not
        if gt_lane_3d.shape[0] < 2 or np.sum(np.logical_and(gt_lane_3d[:, 0] > self.x_min,
                                                            gt_lane_3d[:, 0] < self.x_max)) < 2:
            return -1, np.array([]), np.array([]), np.array([])

        # only keep the portion y is monotonically increasing above a threshold, to prune those super close points
        gt_lane_3d = make_lane_y_mono_inc(gt_lane_3d)
        if gt_lane_3d.shape[0] < 2:
            return -1, np.array([]), np.array([]), np.array([])

        # ignore GT ends before y_ref, for those start at y > y_ref, use its interpolated value at y_ref for association
        # if gt_lane_3d[0, 1] > self.y_ref or gt_lane_3d[-1, 1] < self.y_ref:
        if gt_lane_3d[-1, 1] < self.y_ref:
            return -1, np.array([]), np.array([]), np.array([])

        # resample ground-truth laneline at anchor y steps
        x_values, z_values, visibility_vec = resample_laneline_in_y(gt_lane_3d, self.anchor_y_steps, out_vis=True)

        #差值之后，再次筛选出 可见值个数大于2 的点
        if np.sum(visibility_vec) < 2:
            return -1, np.array([]), np.array([]), np.array([])

        # decide association at visible offset locations
        ass_id = np.argmin(np.linalg.norm(np.multiply(self.anchor_grid_x - x_values, visibility_vec), axis=1))
        # compute offset values
        x_off_values = x_values - self.anchor_grid_x[ass_id]

        return ass_id, x_off_values, z_values, visibility_vec

    def filter_lane(self, lane):
        assert lane[-1][1] <= lane[0][1]
        filtered_lane = []
        used = set()
        for p in lane:
            if p[1] not in used:
                filtered_lane.append(p)
                used.add(p[1])

        return filtered_lane

    # def transform_annotation(self, anno, category, img_wh):
    #     # net size: self.h_net, self.w_net
    #     img_w, img_h = img_wh

    #     old_lanes = anno.copy()

    #     # removing lanes with less than 2 points
    #     old_lanes = filter(lambda x: len(x) > 1, old_lanes)
    #     # sort lane points by Y (bottom to top of the image)
    #     old_lanes = [sorted(lane, key=lambda x: -x[1]) for lane in old_lanes]
    #     # remove points with same Y (keep first occurrence)
    #     old_lanes = [self.filter_lane(lane) for lane in old_lanes]
    #     # normalize the annotation coordinates
    #     old_lanes = [[[x * self.w_net / float(img_w), y * self.h_net / float(img_h)] for x, y in lane]
    #                  for lane in old_lanes]
    #     # create tranformed annotations
    #     # lanes = np.ones((self.max_lanes, 2 + 1 + 1 + 1 + self.n_offsets),
    #     #                 dtype=np.float32) * -1e5  # 2 scores, 1 start_y, 1 start_x, 1 length, S+1 coordinates

    #     # num_category scores, 1 start_y, 1 start_x, S coordinates, S visiblity
    #     lanes = np.ones((self.max_lanes, self.num_category + 1 + 1 + 2 * self.n_offsets),
    #                     dtype=np.float32) * -1e5
    #     # lanes are invalid and all points are invisible by default
    #     lanes[:, 0] = 1
    #     # lanes[:, 1] = 0
    #     lanes[:, 1:self.num_category] = 0
    #     lanes[:, self.num_category + 2 + self.n_offsets:] = 0
    #     for lane_idx, lane in enumerate(old_lanes):
    #         try:
    #             # xs_outside_image, xs_inside_image = self.sample_lane(lane, self.offsets_ys)
    #             xs_outside_image, xs_inside_image, interp_xs_length, extrap_ys_length = \
    #                 self.sample_lane(lane, self.offsets_ys)
    #         except AssertionError:
    #             # print("Sample lane error with #{} lane".format(lane_idx))
    #             continue
    #         if len(xs_inside_image) == 0:
    #             continue
    #         all_xs = np.hstack((xs_outside_image, xs_inside_image))
    #         # lanes[lane_idx, 0] = 0
    #         # lanes[lane_idx, 1] = 1
    #         # lanes[lane_idx, 2] = len(xs_outside_image) / self.n_strips
    #         # lanes[lane_idx, 3] = xs_inside_image[0]
    #         # lanes[lane_idx, 4] = len(xs_inside_image)
    #         # lanes[lane_idx, 5:5 + len(all_xs)] = all_xs

    #         lanes[lane_idx, 0] = 0
    #         lanes[lane_idx, category[lane_idx]] = 1
    #         lanes[lane_idx, self.num_category] = len(xs_outside_image) / self.n_strips
    #         lanes[lane_idx, self.num_category + 1] = xs_inside_image[0]
    #         lanes[lane_idx, self.num_category + 2: self.num_category + 2 + len(all_xs)] = all_xs
    #         # print("extrap_ys_length: ", extrap_ys_length)
    #         # print("len of xs_inside_image: ", len(xs_inside_image))
    #         lanes[lane_idx, self.num_category + 2 + self.n_offsets + extrap_ys_length:
    #                         self.num_category + 2 + self.n_offsets +
    #                         min(len(all_xs), self.n_offsets)] = 1
    #     new_anno = lanes.copy()
    #     # print("label.size in transform_annotation: ", np.shape(new_anno))
    #     return new_anno

    # def sample_lane(self, points, sample_ys):
    #     # this function expects the points to be sorted
    #     points = np.array(points)
    #     if not np.all(points[1:, 1] < points[:-1, 1]):
    #         raise Exception('Annotaion points have to be sorted')
    #     x, y = points[:, 0], points[:, 1]

    #     # interpolate points inside domain
    #     # BUG remains here: https://github.com/lucastabelini/LaneATT/issues/10
    #     assert len(points) > 1
    #     interp = UnivariateSpline(y[::-1], x[::-1], k=min(3, len(points) - 1))
    #     domain_min_y = y.min()
    #     domain_max_y = y.max()
    #     sample_ys_inside_domain = sample_ys[(sample_ys >= domain_min_y) & (sample_ys <= domain_max_y)]
    #     assert len(sample_ys_inside_domain) > 0
    #     interp_xs = interp(sample_ys_inside_domain)

    #     # extrapolate lane to the bottom of the image with a straight line using the 2 points closest to the bottom
    #     num_points_selected = min(10, np.shape(points)[0])
    #     two_closest_points = points[:num_points_selected]
    #     extrap = np.polyfit(two_closest_points[:, 1], two_closest_points[:, 0], deg=2)
    #     extrap_ys = sample_ys[sample_ys > domain_max_y]
    #     extrap_xs = np.polyval(extrap, extrap_ys)
    #     all_xs = np.hstack((extrap_xs, interp_xs))

    #     interp_xs_length = np.shape(interp_xs)[0]
    #     extrap_ys_length = np.shape(extrap_ys)[0]

    #     # separate between inside and outside points
    #     inside_mask = (all_xs >= 0) & (all_xs < self.w_net)
    #     xs_inside_image = all_xs[inside_mask]
    #     xs_outside_image = all_xs[~inside_mask]

    #     return xs_outside_image, xs_inside_image, interp_xs_length, extrap_ys_length

    # # this is for gt labels
    # # use model.decode to decode 2D lanes
    # def label_to_lanes(self, label):
    #     lanes = []
    #     # print("label.size in label_to_lanes: ", np.shape(label))
    #     for l in label:
    #         # if l[1] == 0:
    #         if l[0] != 0:
    #             continue
    #         # xs = l[5:] / self.w_net
    #         xs = l[self.num_category + 2:self.num_category + 2 + self.n_offsets] / self.w_net
    #         ys = self.offsets_ys / self.h_net
    #         # start = int(round(l[2] * self.n_strips))
    #         start = int(round(l[self.num_category] * self.n_strips))
    #         # length = int(round(l[4]))
    #         l_vis = l[self.num_category + 2 + self.n_offsets:]
    #         idx = int(np.nonzero(l_vis == 1)[0][-1])
    #         start_vis_idx = int(np.nonzero(l_vis == 1)[0][0])
    #         start = start_vis_idx
    #         length = idx - start + 1
    #         # idx = self.num_category+2+2*self.n_offsets - 1
    #         # while l[idx] < 1e-5 and idx > self.num_category+2+self.n_offsets:
    #         #     idx -= 1
    #         # length = int(idx - (self.num_category+2+self.n_offsets) - start + 1)
    #         xs = xs[start:start + length][::-1]
    #         ys = ys[start:start + length][::-1]
    #         xs = xs.reshape(-1, 1)
    #         ys = ys.reshape(-1, 1)
    #         points = np.hstack((xs, ys))
    #         if np.shape(points)[0] < 2:
    #             continue
    #         lanes.append(Lane(points=points))
    #     return lanes


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(transformed_dataset, args):
    # transformed_dataset = LaneDataset(dataset_base_dir, json_file_path, args)
    sample_idx = range(transformed_dataset.n_samples)

    g = torch.Generator()
    g.manual_seed(0)

    # sample_idx = sample_idx[0:len(sample_idx)//args.batch_size*args.batch_size]
    discarded_sample_start = len(sample_idx) // args.batch_size * args.batch_size
    if args.proc_id == 0:
        print("Discarding images:")
    if args.proc_id == 0:
        if hasattr(transformed_dataset, '_label_image_path'):
            print(transformed_dataset._label_image_path[discarded_sample_start: len(sample_idx)])
        else:
            print(len(sample_idx) - discarded_sample_start)
    sample_idx = sample_idx[0: discarded_sample_start]

    if args.dist:
        if args.proc_id == 0:
            print('use distributed sampler')
        if 'standard' in args.dataset_name or 'rare_subset' in args.dataset_name or 'illus_chg' in args.dataset_name:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset, shuffle=True,
                                                                           drop_last=True)
            data_loader = DataLoader(transformed_dataset,
                                     batch_size=args.batch_size,
                                     sampler=data_sampler,
                                     num_workers=args.nworkers,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     worker_init_fn=seed_worker,
                                     generator=g,
                                     drop_last=True)
        else:
            data_sampler = torch.utils.data.distributed.DistributedSampler(transformed_dataset)
            data_loader = DataLoader(transformed_dataset,
                                     batch_size=args.batch_size,
                                     sampler=data_sampler,
                                     num_workers=args.nworkers,
                                     pin_memory=True,
                                     persistent_workers=True,
                                     worker_init_fn=seed_worker,
                                     generator=g)
    else:
        if args.proc_id == 0:
            print("use default sampler")
        data_sampler = torch.utils.data.sampler.SubsetRandomSampler(sample_idx)
        data_loader = DataLoader(transformed_dataset,
                                 batch_size=args.batch_size, shuffle = True)

    if args.dist:
        # print("=========data_sampler========")
        return data_loader, data_sampler

    return data_loader




def resample_laneline_in_y_with_vis(input_lane, y_steps, vis_vec):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert(input_lane.shape[0] >= 2)

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")
    f_vis = interp1d(input_lane[:, 1], vis_vec, fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)
    vis_values = f_vis(y_steps)

    x_values = x_values[vis_values > 0.5]
    y_values = y_steps[vis_values > 0.5]
    z_values = z_values[vis_values > 0.5]
    return np.array([x_values, y_values, z_values]).T





def transform_lane_gflat2g(h_cam, X_gflat, Y_gflat, Z_g):
    """
        Given X coordinates in flat ground space, Y coordinates in flat ground space, and Z coordinates in real 3D ground space
        with projection matrix from 3D ground to flat ground, compute real 3D coordinates X, Y in 3D ground space.

    :param P_g2gflat: a 3 X 4 matrix transforms lane form 3d ground x,y,z to flat ground x, y
    :param X_gflat: X coordinates in flat ground space
    :param Y_gflat: Y coordinates in flat ground space
    :param Z_g: Z coordinates in real 3D ground space
    :return:
    """

    X_g = X_gflat - X_gflat * Z_g / h_cam
    Y_g = Y_gflat - Y_gflat * Z_g / h_cam

    return X_g, Y_g


def compute_3d_lanes_all_category(pred_anchor, dataset, anchor_y_steps, h_cam):
    anchor_dim = dataset.anchor_dim
    num_category = dataset.num_category
    anchor_x_steps = dataset.anchor_grid_x
    # use_default_anchor = dataset.use_default_anchor

    lanelines_out = []
    lanelines_prob = []
    num_y_steps = anchor_y_steps.shape[0]

    # output only the visible portion of lane
    """
        An important process is output lanes in the considered y-range. Interpolate the visibility attributes to 
        automatically determine whether to extend the lanes.
    """
    for j in range(pred_anchor.shape[0]):
        # draw laneline
        x_offsets = pred_anchor[j, :num_y_steps]
        x_g = x_offsets + anchor_x_steps[j]
        z_g = pred_anchor[j, num_y_steps:2 * num_y_steps]
        visibility = pred_anchor[j, 2 * num_y_steps:3 * num_y_steps]
        # visibility = np.ones_like(visibility)
        category = pred_anchor[j, 3 * num_y_steps:]
        if np.argmax(category) == 0:
            continue
        line = np.vstack([x_g, anchor_y_steps, z_g]).T
        # line = line[visibility > prob_th, :]
        # convert to 3D ground space
        x_g, y_g = transform_lane_gflat2g(h_cam, torch.from_numpy(line[:, 0]), torch.from_numpy(line[:, 1]), torch.from_numpy(line[:, 2]))
        line[:, 0] = x_g
        line[:, 1] = y_g
        # print("line before resample: ", line)
        # print("line visibility before resample: ", visibility)
        line = resample_laneline_in_y_with_vis(line, anchor_y_steps, visibility)
        # print("line after resample: ", line)
        if line.shape[0] >= 2:
            lanelines_out.append(line.data.tolist())
            lanelines_prob.append(category.tolist())

    return lanelines_out, lanelines_prob


def unormalize_lane_anchor(anchor, dataset):
    num_y_steps = dataset.num_y_steps
    anchor_dim = dataset.anchor_dim
    for i in range(dataset.num_types):
        anchor[:, i*anchor_dim:i*anchor_dim + num_y_steps] = \
            np.multiply(anchor[:, i*anchor_dim: i*anchor_dim + num_y_steps], dataset._x_off_std)

        anchor[:, i*anchor_dim + num_y_steps: i*anchor_dim + 2*num_y_steps] = \
            np.multiply(anchor[:, i*anchor_dim + num_y_steps: i*anchor_dim + 2*num_y_steps], dataset._z_std)

