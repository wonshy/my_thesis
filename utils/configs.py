import argparse
import os
import os.path as ops
import numpy as np


def config(args):

    
    
    
    # print("======config=======")

    # rank default is 0.
    args.proc_id = 0

    # 3d loss, vis | prob | reg, default 1.0 | 1.0 | 1.0,  suggest 10.0 | 4.0 | 1.0
    # used if not learnable_weight_on
    args.loss_dist = [10.0, 4.0, 1.0]

    # learnable weight
    # in best model setting, they are 10, 4, 1
    # factor = 1 / exp(weight)
    # args.learnable_weight_on = True
    # args._3d_vis_loss_weight = 0.0  # -2.3026
    # args._3d_prob_loss_weight = 0.0  # -1.3863
    # args._3d_reg_loss_weight = 0.0

    # 300 sequence
    args.dataset_name = 'openlane'


    dataset_part = 'lane3d_1000'
    # dataset path
    root_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    dataset_path = os.path.join(root_path, "openlane")
    extend_path=ops.join(dataset_path, 'extend')


    args.dataset_dir = ops.join(dataset_path, 'images/')
    args.data_dir = ops.join(dataset_path, dataset_part)

    args.extend_dataset_dir = ops.join(extend_path, 'images/')
    args.extend_data_dir = ops.join(extend_path, dataset_part)


    # args.extend_dataset_dir = ops.join(dataset_path, 'extend/images/')
    # args.extend_data_dir = ops.join(dataset_path, 'extend/lane3d_1000/')

    # sav path
    args.save_prefix = ops.join(os.getcwd(), 'data_splits')
    args.save_path = ops.join(args.save_prefix, args.dataset_name)

    # this used for Normalization
    args.vgg_mean = [0.485, 0.456, 0.406]
    args.vgg_std = [0.229, 0.224, 0.225]

    # for the case only running evaluation
    #args.evaluate = True
    args.evaluate_case = False


    # settings for save and visualize
    args.print_freq = 50
    # args.save_freq = 50

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


    args.data_aug_conf = {
        #efficient 要求分辨率 必须32的倍数
        # 'final_dim': (128, 352),
        # 'final_dim': (320, 480),#good

        ######B0-good############
        # 'resize_lim': (0.124, 0.135),
        # 'final_dim': (160, 256),#good
        # 'rot_lim': (-5.4, 5.4),
        # 'bot_pct_lim': (0.0, 0.03),
        ######B6-good############
        # 'resize_lim': (0.311, 0.289),
        # 'final_dim': (384, 576),#good
        # 'rot_lim': (-18.4, 18.4),
        # 'bot_pct_lim': (0.01, 0.13),


##################320x480######################################

        # 'resize_lim': (0.238, 0.262),
        # 'final_dim': (320, 480),


##################256x384######################################

        # 'resize_lim': (0.192, 0.241),
        # 'final_dim': (256, 384),
        # 'rot_lim': (-5.4, 5.4),
        # 'bot_pct_lim': (0.0, 0.11),

##################128x192######################################

        'resize_lim': (0.092, 0.108),
        'final_dim': (128, 192),#good
        'rot_lim': (-5.4, 5.4),
        'bot_pct_lim': (0.0, 0.22),


##################640x480######################################

        # 'resize_lim': (0.341, 0.374),
        # 'final_dim': (480, 640),
        # 'rot_lim': (-5.4, 5.4),
        # 'bot_pct_lim': (0.0, 0.11),



        'FRONT_H': 1280, 'FRONT_W': 1920,
        'SIDE_H': 886, 'SIDE_W': 1920,

        'rand_flip': True,

        # src=(h:1280, w:1920)
        # dest = src*rot_limt*(1 - bot_pct_lim)
        # crop =(dest - final_dim)
        # img =  (crop_w, crop_h, crop_w + final_dim_W, crop_h + final_dim_H)
        # 最终的结果是，越大的裁剪的越多， 填补的也越多

        'cams': {'CAM_FRONT':0, 'CAM_FRONT_LEFT':1, 'CAM_FRONT_RIGHT':2,
                    'CAM_LEFT':3, 'CAM_RIGHT':4},
        'cams_sel': ['CAM_FRONT','CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_LEFT','CAM_RIGHT' ],
        # 'cams_sel': ['CAM_FRONT' ],

    }



    # origh .
    args.org_h = 1280
    args.H = args.org_h

    args.org_w = 1920
    args.W = args.org_w

    # resize
    # args.resize_h = 360
    # args.resize_w = 480
    # follow lift-split



    # args.resize_h = 128
    # args.resize_w = 352

    # # ipm
    # args.ipm_h = 208
    args.ipm_w = 128

    # args.crop_y = 0
    args.no_centerline = True
    args.no_3d = False
    args.fix_cam = False
    args.pred_cam = False

    # # Placeholder, shouldn't be used
    # args.K = np.array([[1000., 0., 960.],
    #                     [0., 1000., 640.],
    #                     [0., 0., 1.]])

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

    # args.max_grad_norm = 5.0
    # args.pos_weight = 2.13
    # args.logdir = './runs'

    # args.xbound = [-50.0, 50.0, 0.5]
    # args.ybound = [-50.0, 50.0, 0.5]
    # args.zbound = [-10.0, 10.0, 20.0]
    # args.dbound = [4.0, 45.0, 1.0]

    
    # args.xbound = [-224.0, 224.0, 0.5]
    # args.ybound = [-52.0, 52.0, 0.5]
    # args.zbound = [-10.0, 10.0, 20.0]
    # args.dbound = [4.0, 45.0, 1.0]

################### nornal 
    # args.xbound = [-56.0, 56.0, 1.0]
    # args.ybound = [-13.0, 13.0, 1.0]

# ################### X2 
#     args.xbound = [-56.0, 56.0, 0.5]
#     args.ybound = [0.0, 52.0, 1.0]

# ################### X4
#     # args.xbound = [-112.0, 112.0, 0.5]
#     # args.ybound = [0.0, 104.0, 1.0]

#     args.zbound = [-10.0, 10.0, 20.0]
#     args.dbound = [4.0, 45.0, 1.0]




    # args.xbound = [-28.0, 28.0, 0.5]#56
    # args.ybound = [0.0, 104.0, 1] #52   104

    # args.zbound = [-10.0, 10.0, 20.0]
    # args.dbound = [4.0, 104.0, 1]

############################################################################

    # args.xbound = [-14.0, 14.0, 0.25]#56
    # args.ybound = [-1.0, args.top_view_region[0, 1].astype(float), 0.5] #52   104

    # args.xbound = [-14.0, 14.0, 0.03125]#56
    # args.ybound = [3.0, 107.0, 0.5] #52   104

    args.xbound = [-14.0, 14.0, 0.25]#56
    args.ybound = [3.0, 107.0, 0.5] #52   104

    args.zbound = [-10.0, 10.0, 20.0]
    args.dbound = [2.0, args.top_view_region[0, 1].astype(float), 1.0]
############################################################################



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

    if args.evaluate_flops or args.evaluate_fps:
        args.batch_size = 1
    else:
        #b3 -> 4, b0 -> 9
        #args.batch_size = 3
        args.batch_size = 2

    

    args.nworkers = args.batch_size

    # args.bsz = args.batch_size

    # args.lr = 1e-3
    # args.weight_decay = 1e-7

    args.world_size = 4