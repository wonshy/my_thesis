import argparse
import os
import os.path as ops
import numpy as np


def config(args):

    args.use_slurm = False
    
    
    
    # print("======config=======")

    # rank default is 0.
    args.proc_id = 0

    # 3d loss, vis | prob | reg, default 1.0 | 1.0 | 1.0,  suggest 10.0 | 4.0 | 1.0
    # used if not learnable_weight_on
    args.loss_dist = [10.0, 4.0, 1.0]

    # learnable weight
    # in best model setting, they are 10, 4, 1
    # factor = 1 / exp(weight)
    args.learnable_weight_on = True
    args._3d_vis_loss_weight = 0.0  # -2.3026
    args._3d_prob_loss_weight = 0.0  # -1.3863
    args._3d_reg_loss_weight = 0.0

    # 300 sequence
    args.dataset_name = 'openlane'

    # dataset path
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    dataset_path = os.path.join(root_path, "openlane")
    args.dataset_dir = ops.join(dataset_path, 'images/')
    args.data_dir = ops.join(dataset_path, 'lane3d_1000/')
    args.extend_dataset_dir = ops.join(dataset_path, 'extend/images/')
    args.extend_data_dir = ops.join(dataset_path, 'extend/lane3d_1000/')

    # sav path
    args.save_prefix = ops.join(os.getcwd(), 'data_splits')
    args.save_path = ops.join(args.save_prefix, args.dataset_name)

    # this used for Normalization
    args.vgg_mean = [0.485, 0.456, 0.406]
    args.vgg_std = [0.229, 0.224, 0.225]

    # for the case only running evaluation
    args.evaluate = False
    args.evaluate_case = False


    # settings for save and visualize
    args.print_freq = 50
    args.save_freq = 50

    args.save_json_path = "data_splits/"


    args.S = 72

    # args.mod ='PersFormer'
    args.mod ='paper'
    dataroot = '/home/wucunyin/workspaces/paper/Dataset'
    # 训练的次数
    args.nepochs = 10000
    args.gpuid = 0
    args.start_epoch = 0
    args.pretrained = False


    # origh .
    args.org_h = 1280
    args.H = args.org_h

    args.org_w = 1920
    args.W = args.org_w

    # resize
    # args.resize_h = 360
    # args.resize_w = 480
    # follow lift-split
    args.resize_h = 128
    args.resize_w = 352

    # ipm
    args.ipm_h = 208
    args.ipm_w = 128

    args.crop_y = 0
    args.no_centerline = True
    args.no_3d = False
    args.fix_cam = False
    args.pred_cam = False

    # Placeholder, shouldn't be used
    args.K = np.array([[1000., 0., 960.],
                        [0., 1000., 640.],
                        [0., 0., 1.]])

    # ddp related
    args.dist = True 
    args.sync_bn = True

    # specify model settings
    args.top_view_region = np.array([[-10, 103], [10, 103], [-10, 3], [10, 3]])
    args.anchor_y_steps = np.array([5, 10, 15, 20, 30, 40, 50, 60, 80, 100])
    args.num_y_steps = len(args.anchor_y_steps)

    # TODO: constrain max lanes in gt
    args.max_lanes = 20
    args.num_category = 21

    args.prob_th = 0.5
    args.num_class = 2  # 1 background + n lane labels
    args.y_ref = 5  # new anchor prefer closer range gt assign

    args.use_default_anchor = False

    args.max_grad_norm = 5.0
    args.pos_weight = 2.13
    args.logdir = './runs'

    # args.xbound = [-50.0, 50.0, 0.5]
    # args.ybound = [-50.0, 50.0, 0.5]
    # args.zbound = [-10.0, 10.0, 20.0]
    # args.dbound = [4.0, 45.0, 1.0]

    # args.xbound = [-52.0, 52.0, 0.5]
    # args.ybound = [-224.0, 224.0, 0.5]
    args.ybound = [-52.0, 52.0, 0.5]
    args.xbound = [-224.0, 224.0, 0.5]
# 480， 320
# 112
# 26

    args.zbound = [-10.0, 10.0, 20.0]
    args.dbound = [4.0, 45.0, 1.0]

    # optimizer
    args.optimizer = 'adam'
    args.learning_rate = 2e-4
    args.weight_decay = 0.001

    # grad clip
    args.clip_grad_norm = 35.0
    args.loss_threshold = 1e5

    # scheduler
    args.lr_policy = "cosine"
    args.T_max = 8
    args.eta_min = 1e-5

    #        args.batch_size = 2
    args.batch_size = 1

    args.bsz = args.batch_size

    args.nworkers = 4
    #args.nworkers = 4

    args.lr = 1e-3
    args.weight_decay = 1e-7

    args.world_size = 4
    #args.nworkers = 4
