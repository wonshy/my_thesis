import argparse
import os
import torch
import time

from nms import nms


from tqdm import tqdm
from data.tools import *
from data.load_data import *
from model.network import *
from scipy.special import softmax
from utils.eval_lane import *
from utils import *
import shutil


# ddp related
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .ddp import *

from tensorboardX import SummaryWriter



class Logger(object):
    """
    Source https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        self.fpath = fpath
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()




class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def nms_bev(batch_output_net, args):

    #应用nms  -   对不同锚点 对同一GT的预测 进行过滤
    """apply nms to filter predictions of same GT from different anchors"""

    #anchor 的维度 x, z, vision, category
    anchor_dim = 3 * args.num_y_steps + args.num_category
    #anchor 的开始位置x，这个在前面绘制锚的过程中已经形成。
    anchor_x_steps = args.anchor_grid_x

    batch_output_net = batch_output_net.reshape(batch_output_net.shape[0], batch_output_net.shape[1], anchor_dim)
    # print("cate before softmax: ", batch_output_net[:, :, anchor_dim-args.num_category:].shape)
    # print(batch_output_net[0, :16, :])

    #车道线分类部分执行softmax之后，由之前的数值归一化到概率。
    batch_output_net[:, :, anchor_dim - args.num_category:] = \
        softmax(batch_output_net[:, :, anchor_dim - args.num_category:], axis=2)
    # print("cate after softmax: ", batch_output_net[:, :, anchor_dim-args.num_category:].shape)
    # print(batch_output_net[0, :16, :])

    #针对具体的每一个 batch 进行处理
    for i, output_net in enumerate(batch_output_net):
        # pack prediction data to nms library format
        scores = torch.zeros(len(output_net)).cuda()
        output_net_nms = torch.zeros(len(output_net), 2 + 3 + args.S).cuda()
        pre_nms_valid_anchor_id = []
        valid_count = 0

        #针对具体的 anchor 进行处理
        for j, output_anchor in enumerate(output_net):
            # print("max cate id: ", np.argmax(output_anchor[anchor_dim-args.num_category:]))
            #取出大于visible阈值的位置值，
            visible_yid = np.where(output_anchor[2 * args.num_y_steps: 3 * args.num_y_steps] > args.prob_th)[0]
            # print("vis points: ", len(visible_yid))

            #如果可见点的数量少于2，或者最大值的导引号等于category 第1个点则忽略( 为啥category第一个导引点就不能为最大的呢？)。
            if np.argmax(output_anchor[anchor_dim - args.num_category:]) == \
                    anchor_dim - args.num_category or len(visible_yid) < 2:
                # print("skip")
                continue

            #存入需要nms的有效anchor
            pre_nms_valid_anchor_id.append(j)

            # 筛选出来最大值，并将其填入到scores
            #add numpy transe.    every anchor scores.
            scores[valid_count] = np.max(output_anchor[anchor_dim - args.num_category + 1:]).item()

            yid_start = visible_yid[0]
            # approximation here, since there can be invisible points in between
            yid_num = visible_yid[-1] - visible_yid[0] + 1
            output_net_nms[valid_count, 2] = (yid_start / (args.S - 1)).item()
            output_net_nms[valid_count, 4] = yid_num.item()
            # 将相对x的开始位置 + 预测的相对位置， 进行在5 , 5+10（num_y_steps）之间 赋x的值。
            # 也就是该anchor的所有点（10个点），使用绝对的长度。
            output_net_nms[valid_count, 5: 5 + args.num_y_steps] = \
                     torch.from_numpy(output_anchor[:args.num_y_steps] + anchor_x_steps[j])

            valid_count = valid_count + 1
        scores = scores[:valid_count]
        output_net_nms = output_net_nms[:valid_count]
        # print("scores: ", scores)
        # print(output_net_nms[:, :16])


        keep, num_to_keep, _ = nms(output_net_nms, scores, overlap=args.nms_thres_3d, top_k=args.max_lanes)
        # print("keep before reduction: ", keep)
        keep = keep[:num_to_keep].tolist()
        # print("keep after reduction: ", keep)
        for jj, anchor_id in enumerate(pre_nms_valid_anchor_id):
            if jj not in keep:
                # update category as invalid, so that it can be filted in compute_3d_lanes()
                batch_output_net[i][anchor_id][anchor_dim - args.num_category] = 1.0

    return batch_output_net



# trick from stackoverflow
def str2bool(argument):
    if argument.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif argument.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Wrong argument in argparse, should be a boolean')

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def define_args():
    parser = argparse.ArgumentParser(description='paper work')
    # Paths settings
    parser.add_argument('--data_dir', type=str, help='The path of dataset json files (annotations)')
    parser.add_argument('--dataset_dir', type=str, help='The path of dataset image files (images)')
    parser.add_argument('--save_path', type=str, default='data_splits/', help='directory to save output')



    # DDP setting
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--gpu', type=int, default = 0)
    parser.add_argument('--world_size', type=int, default = 1)
    
    parser.add_argument('--no_cuda', action='store_true', help='if gpu available')
    parser.add_argument("--no_tb", type=str2bool, nargs='?', const=True, default=False, help="Use tensorboard logging by tensorflow")


    parser.add_argument('--nms_thres_3d', type=float, default=1.0, help='nms threshold to filter detections in BEV, unit: meter')



    return parser

def first_run(save_path):
    txt_file = os.path.join(save_path,'first_run.txt')
    if not os.path.exists(txt_file):
        open(txt_file, 'w').close()
    else:
        saved_epoch = open(txt_file).read()
        if saved_epoch is None:
            print('You forgot to delete [first run file]')
            return '' 
        return saved_epoch
    return ''


class Runner:
    def __init__(self, args):
        
        
        self.args = args
        
        # Check GPU availability
        if args.proc_id == 0:
            if not args.no_cuda and not torch.cuda.is_available():
                raise Exception("No gpu available for usage")
            if torch.cuda.device_count() >= 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                torch.cuda.empty_cache()


        self.grid_conf = {
            'xbound': args.xbound,
            'ybound': args.ybound,
            'zbound': args.zbound,
            'dbound': args.dbound,
        }
        self.data_aug_conf = {
            'resize_lim': (0.193, 0.225),
            'final_dim': (128, 352),
            'rot_lim': (-5.4, 5.4),
            'H': 1280, 'W': 1920,
            'rand_flip': True,
            'bot_pct_lim': (0.0, 0.22),
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                     'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
            'Ncams': 1,
        }

        #set device.
        self.device = torch.device("cuda", args.local_rank)


        # save_id = args.mod
        # args.save_json_path = args.save_path
        # args.save_path = os.path.join(args.save_path, save_id)
        # if args.proc_id == 0:
        #     mkdir_if_missing(args.save_path)
        #     mkdir_if_missing(os.path.join(args.save_path, 'example/'))
        #     mkdir_if_missing(os.path.join(args.save_path, 'example/train'))
        #     mkdir_if_missing(os.path.join(args.save_path, 'example/valid'))

        # Get Dataset
        if args.proc_id == 0:
            print("Loading Dataset ...")

            
        # Get Dataset
        self.val_gt_file = os.path.join(args.save_path, 'test.json')
        # TODO:GPU need?
        # self.train_dataset, self.train_loader, self.train_sampler = self._get_train_dataset()
        # # self.valid_dataset, self.valid_loader, self.valid_sampler = self._get_valid_dataset()

        self.train_dataset, self.train_loader, self.train_sampler = self._get_train_dataset()
        self.valid_dataset, self.valid_loader, self.valid_sampler  = self._get_valid_dataset()

        self.evaluator = LaneEval(args)

        # self.model = compile_model(self.grid_conf, self.data_aug_conf, 1, args.num_y_steps, args.num_category)


        # self.valid_dataset, self.valid_loader, self.valid_sampler = self._get_valid_dataset()
        self.criterion = Laneline_loss_gflat_multiclass(self.train_dataset.num_types, args.num_y_steps,
                                                        args.pred_cam, args.num_category,
                                                        args.loss_dist)

        # Tensorboard writer
        if not args.no_tb and args.proc_id == 0:
            tensorboard_path = os.path.join(args.save_path, 'Tensorboard/')
            mkdir_if_missing(tensorboard_path)
            self.writer = SummaryWriter(tensorboard_path)


    def _get_train_dataset(self):
        args = self.args
        train_dataset = lane_dataset(args.dataset_dir, args.data_dir + 'training/', 
                    args.extend_dataset_dir, args.extend_data_dir + 'training/',
                    args, data_aug=True,save_std=True)

        # TODO:GPU need?
        # # train_dataset.normalize_lane_label()
        # train_loader, train_sampler = get_loader(train_dataset, args)
        # return train_dataset, train_loader, train_sampler

        # train_dataset.normalize_lane_label()
        train_loader, data_sampler = get_loader(train_dataset, args)
        return train_dataset, train_loader, data_sampler
    def _get_valid_dataset(self):

        args = self.args

        valid_dataset = lane_dataset(args.dataset_dir, args.data_dir + 'validation/',
                    args.extend_dataset_dir, args.extend_data_dir + 'validation/', args)

        # assign std of valid dataset to be consistent with train dataset
        valid_dataset.set_x_off_std(self.train_dataset._x_off_std)
        valid_dataset.set_z_std(self.train_dataset._z_std)
        # valid_dataset.normalize_lane_label()
        valid_loader, valid_sampler = get_loader(valid_dataset, args)

        return valid_dataset, valid_loader, valid_sampler


    def _get_model_ddp(self):
        args = self.args
        # Define network
        model = compile_model(self.grid_conf, self.data_aug_conf, 1, args.num_y_steps, args.num_category)

        if args.sync_bn:
            if args.proc_id == 0:
                print("Convert model with Sync BatchNorm")
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if not args.no_cuda:
            # Load model on gpu before passing params to optimizer
            device = torch.device("cuda", args.local_rank)
            model = model.to(device)

        """
            first load param to model, then model = DDP(model)
        """
        # Logging setup
        best_epoch = 0
        lowest_loss = np.inf
        best_f1_epoch = 0
        best_val_f1 = -1e-5
        optim_saved_state = None
        schedule_saved_state = None

        # resume model
        args.resume = first_run(args.save_path)
        if args.resume:
            model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, \
                optim_saved_state, schedule_saved_state = self.resume_model(args, model)
        elif args.pretrained and args.proc_id == 0:
            path = 'models/pretrain/model_pretrain.pth.tar'
            if os.path.isfile(path):
                checkpoint = torch.load(path)
                model.load_state_dict(checkpoint['state_dict'])
                print("Use pretrained model in {} to start training".format(path))
            else:
                raise Exception("No pretrained model found in {}".format(path))

        dist.barrier()
        # DDP setting
        if args.distributed:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        # Define optimizer and scheduler
        '''
            Define optimizer after DDP init
        '''
        optimizer = define_optim(args.optimizer, model.parameters(),
                                args.learning_rate, args.weight_decay)
        scheduler = define_scheduler(optimizer, args)

        # resume optimizer and scheduler
        if optim_saved_state is not None:
            print("proc_id-{} load optim state".format(args.proc_id))
            optimizer.load_state_dict(optim_saved_state)
        if schedule_saved_state is not None:
            print("proc_id-{} load scheduler state".format(args.proc_id))
            scheduler.load_state_dict(schedule_saved_state)

        return model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1

    def resume_model(self, args, model):
        path = os.path.join(args.save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(int(args.resume)))
        if os.path.isfile(path):
            print("=> loading checkpoint from {}".format(path))
            checkpoint = torch.load(path, map_location='cpu')
            if args.proc_id == 0:
                log_file_name = 'log_train_start_{}.txt'.format(args.resume)
                # Redirect stdout
                sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
                model.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_epoch = checkpoint['best_epoch']
            lowest_loss = checkpoint['lowest_loss']
            best_f1_epoch = checkpoint['best_f1_epoch']
            best_val_f1 = checkpoint['best_val_f1']
            optim_saved_state = checkpoint['optimizer']
            schedule_saved_state = checkpoint['scheduler']
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if args.proc_id == 0:
                log_file_name = 'log_train_start_0.txt'
                # Redirect stdout
                sys.stdout = Logger(os.path.join(args.save_path, log_file_name))
                print("=> Warning: no checkpoint found at '{}'".format(path))
            best_epoch = 0
            lowest_loss = np.inf
            best_f1_epoch = 0
            best_val_f1 = -1e-5
            optim_saved_state = None
            schedule_saved_state = None
        return model, best_epoch, lowest_loss, best_f1_epoch, best_val_f1, optim_saved_state, schedule_saved_state


    def reduce_all_loss(self, args, loss_list, loss, loss_3d_dict, num):
        reduced_loss = loss.data
        reduced_loss_all = reduce_tensors(reduced_loss, world_size=args.world_size)
        losses = loss_list[0]
        losses.update(to_python_float(reduced_loss_all), num)

        reduced_vis_loss = loss_3d_dict['vis_loss'].data
        reduced_vis_loss = reduce_tensors(reduced_vis_loss, world_size=args.world_size)
        losses_3d_vis = loss_list[1]
        losses_3d_vis.update(to_python_float(reduced_vis_loss), num)

        reduced_prob_loss = loss_3d_dict['prob_loss'].data
        reduced_prob_loss = reduce_tensors(reduced_prob_loss, world_size=args.world_size)
        losses_3d_prob = loss_list[2]
        losses_3d_prob.update(to_python_float(reduced_prob_loss), num)

        reduced_reg_loss = loss_3d_dict['reg_loss'].data
        reduced_reg_loss = reduce_tensors(reduced_reg_loss, world_size=args.world_size)
        losses_3d_reg = loss_list[3]
        losses_3d_reg.update(to_python_float(reduced_reg_loss), num)

        return loss_list


    def train(self):
        args = self.args

        #Note：it ussing.
        best_val_f1 = 0
        lowest_loss = 0

        # Get Dataset
        train_dataset = self.train_dataset
        train_loader = self.train_loader
        train_sampler = self.train_sampler
        # Define model or resume

        self.model, optimizer, scheduler, best_epoch, lowest_loss, best_f1_epoch, best_val_f1  = self._get_model_ddp()
        criterion = self.criterion

        #put criterion data to cuda.
        criterion = self.criterion
        if not args.no_cuda:
            device = torch.device("cuda", args.local_rank)
            criterion = criterion.to(device)
        bceloss = nn.BCEWithLogitsLoss()

        # # Print model basic info
        # if args.proc_id == 0:
        #     print(40*"="+"\nArgs:{}\n".format(args)+40*"=")
        #     print("Init model: '{}'".format(args.mod))
        #     print("Number of parameters in model {} is {:.3f}M".format(args.mod, sum(tensor.numel() for tensor in model.parameters())/1e6))

        if args.proc_id == 0:
            writer = self.writer

        # Start training and validation for nepochs
        for epoch in range(args.start_epoch, args.nepochs):
            if args.proc_id == 0:
                print("\n => Start train set for EPOCH {}".format(epoch + 1))
                lr = optimizer.param_groups[0]['lr']
                print('lr is set to {}'.format(lr))

            if args.distributed:
                train_sampler.set_epoch(epoch)

            #分割，没啥用
            # if epoch > args.seg_start_epoch:
            #     args.loss_seg_weight = 10.0


            # Define container objects to keep track of multiple losses/metrics
            batch_time = AverageMeter()
            data_time = AverageMeter()          # compute FPS
            losses = AverageMeter()
            losses_3d_vis = AverageMeter()
            losses_3d_prob = AverageMeter()
            losses_3d_reg = AverageMeter()

            # Specify operation modules
            self.model.train()

            # compute timing
            end = time.time()
            for i, (
                    idx_json_file, image, gt_anchor, idx, intrinsics, extrinsics, aug_mat,
                    rots, trans, images, all_intrinsics, all_extrinsics,all_rots, all_trans
            ) in tqdm(enumerate(train_loader)):
                # Time dataloader
                data_time.update(time.time() - end)

                # Put data on gpu if possible
                if not args.no_cuda:
                    image, gt_anchor = image.cuda(non_blocking=True), gt_anchor.cuda(non_blocking=True)
                    intrinsics = intrinsics.cuda()
                    intrinsics = intrinsics.cuda()
                    rots = rots.cuda()
                    trans = trans.cuda()
                    
                    all_rots = all_rots.cuda()
                    all_trans = all_trans.cuda()
                    all_intrinsics = all_intrinsics.cuda()

                image = image.contiguous().float()
                # # Run model
                optimizer.zero_grad()

                # preds = self.model(image,
                #                    rots, trans,
                #                    intrinsics
                #                    )


                preds = self.model(images,
                                   all_rots, all_trans,
                                   all_intrinsics
                                   )

                loss, loss_3d_dict = criterion(preds, gt_anchor)
                # print(loss_3d_dict)

                # Clip gradients (usefull for instabilities or mistakes in ground truth)
                if args.clip_grad_norm != 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), args.clip_grad_norm)

                # Setup backward pass
                loss.backward()

                # update params
                optimizer.step()

                #这里重点查看
                # reduce loss from all gpu, then update losses
                loss_list = [losses, losses_3d_vis, losses_3d_prob, losses_3d_reg]
                loss_list = self.reduce_all_loss(args, loss_list, loss, loss_3d_dict, image.size(0))

                # Time trainig iteration
                batch_time.update(time.time() - end)
                end = time.time()

                # Print info
                if (i + 1) % args.print_freq == 0 and args.proc_id == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(epoch+1, i+1, len(train_loader), 
                                            batch_time=batch_time, data_time=data_time, loss=loss_list[0]))


            # Adjust learning rate
            scheduler.step()

            # loss terms need to be all reduced, eval_stats need to be all gather
            # Do them all in validate
            loss_valid_list, eval_stats = self.validate(self.model, epoch, vis=True) #vis ????????????????
            total_score = loss_list[0].avg

            if args.proc_id == 0:
                # File to keep latest epoch
                with open(os.path.join(args.save_path, 'first_run.txt'), 'w') as f:
                    f.write(str(epoch + 1))
                # Save model
                to_save = False
                if total_score < lowest_loss:
                    best_epoch = epoch + 1
                    lowest_loss = total_score
                if eval_stats[0] > best_val_f1:
                    to_save = True
                    best_f1_epoch = epoch + 1
                    best_val_f1 = eval_stats[0]

                self.crit_string = "loss_gflat_multiclass"    
                # print validation result every epoch 
                print("===> Average {}-loss on training set is {:.8f}".format(self.crit_string, loss_list[0].avg))
                print("===> Average {}-loss on validation set is {:.8f}".format(self.crit_string, loss_valid_list[0].avg))
                print("===> Evaluation laneline F-measure: {:.8f}".format(eval_stats[0]))
                print("===> Evaluation laneline Recall: {:.8f}".format(eval_stats[1]))
                print("===> Evaluation laneline Precision: {:.8f}".format(eval_stats[2]))
                print("===> Evaluation laneline Category Accuracy: {:.8f}".format(eval_stats[3]))
                print("===> Evaluation laneline x error (close): {:.8f} m".format(eval_stats[4]))
                print("===> Evaluation laneline x error (far): {:.8f} m".format(eval_stats[5]))
                print("===> Evaluation laneline z error (close): {:.8f} m".format(eval_stats[6]))
                print("===> Evaluation laneline z error (far): {:.8f} m".format(eval_stats[7]))
                print("===> Last best {}-loss was {:.8f} in epoch {}".format(self.crit_string, lowest_loss, best_epoch))
                print("===> Last best F1 was {:.8f} in epoch {}".format(best_val_f1, best_f1_epoch))

                self.save_checkpoint({
                    'arch': args.mod,
                    'state_dict': self.model.module.state_dict(),
                    'epoch': epoch + 1,
                    'loss': total_score,
                    'f1': eval_stats[0],
                    'best_epoch': best_epoch,
                    'lowest_loss': lowest_loss,
                    'best_f1_epoch': best_f1_epoch,
                    'best_val_f1': best_val_f1,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()}, to_save, epoch+1, args.save_path)

            dist.barrier()
            torch.cuda.empty_cache()
        
        # at the end of training
        if not args.no_tb and args.proc_id == 0:
            writer.close()


    def save_checkpoint(self, state, to_copy, epoch, save_path):
        filepath = os.path.join(save_path, 'checkpoint_model_epoch_{}.pth.tar'.format(epoch))
        torch.save(state, filepath)
        if to_copy:
            if epoch > 1:
                lst = glob.glob(os.path.join(save_path, 'model_best*'))
                if len(lst) != 0:
                    os.remove(lst[0])
            shutil.copyfile(filepath, os.path.join(save_path, 'model_best_epoch_{}.pth.tar'.format(epoch)))
            print("Best model copied")
        #add by wucunyin. just save best checkpoint.
        os.remove(filepath)

    def validate(self, model, epoch=0, vis=False):
        args = self.args
        loader = self.valid_loader
        dataset = self.valid_dataset
        criterion = self.criterion

        if not args.no_cuda:
            device = torch.device("cuda", args.local_rank)
            criterion = criterion.to(device)
        bceloss = nn.BCEWithLogitsLoss()
        
        # vs_saver = self.vs_saver
        val_gt_file = self.val_gt_file
        # valid_set_labels = self.valid_set_labels
        # Define container to keep track of metric and loss
        losses = AverageMeter()
        losses_3d_vis = AverageMeter()
        losses_3d_prob = AverageMeter()
        losses_3d_reg = AverageMeter()

        pred_lines_sub = []
        gt_lines_sub = []

        # Evaluate model
        model.eval()

        # Start validation loop
        with torch.no_grad():

            for i, (
                    json_files, image, gt_anchor, idx, intrinsics, extrinsics,
                    rots, trans, images, all_intrinsics, all_extrinsics, all_rots, all_trans
            ) in tqdm(enumerate(loader)):
            
                if not args.no_cuda:
                    image, gt_anchor = image.cuda(non_blocking=True), gt_anchor.cuda(non_blocking=True)
                    gt_intrinsic = intrinsics.cuda()
                    extrinsics = extrinsics.cuda()
                    rots = rots.cuda()
                    trans = trans.cuda()
                    all_rots = all_rots.cuda()
                    all_trans = all_trans.cuda()
                    all_intrinsics = all_intrinsics.cuda()

                image = image.contiguous().float()

               # Inference model
                # preds = self.model(image,
                #                    rots, trans,
                #                    intrinsics
                #                    )
                preds = self.model(images,
                    all_rots, all_trans,
                    all_intrinsics
                    )

                loss, loss_3d_dict = criterion(preds, gt_anchor)
                # losses += loss


                # print(loss)

                # reduce loss from all gpu, then update losses
                loss_list = [losses, losses_3d_vis, losses_3d_prob, losses_3d_reg]
                loss_list = self.reduce_all_loss(args, loss_list, loss, loss_3d_dict, image.size(0))

                # Print info
                if (i + 1) % args.print_freq == 0 and args.proc_id == 0:
                        print('Test: [{0}/{1}]\t'
                                'Loss {loss.val:.8f} ({loss.avg:.8f})'.format(i+1, len(loader), loss=loss_list[0]))



                intrinsics = torch.squeeze(intrinsics, 1)
                extrinsics = torch.squeeze(extrinsics, 1)

                intrinsics = intrinsics.data.cpu().numpy()
                extrinsics = extrinsics.data.cpu().numpy()
                preds = preds.data.cpu().numpy()
                gt_anchor = gt_anchor.data.cpu().numpy()

                # unormalize lane outputs, 这里为啥使用标准差去乘以每个数？
                num_el = image.size(0)
                for j in range(num_el):
                    unormalize_lane_anchor(preds[j], dataset)
                    unormalize_lane_anchor(gt_anchor[j], dataset)



                # Apply nms on network BEV output
                if not args.use_default_anchor:
                    preds = nms_bev(preds, args)


                # Write results
                for j in range(num_el):
                    im_id = idx[j]
                    # saving json style
                    # json_line = valid_set_labels[im_id]
                    json_file = json_files[j]
                    with open(json_file, 'r') as file:
                        file_lines = [line for line in file]
                        json_line = json.loads(file_lines[0])

                    gt_lines_sub.append(copy.deepcopy(json_line))

                    lane_anchors = preds[j]
                    # convert to json output format  , laneLines是可见x,y,z值，laneLines_prob，是对应得车道线的策略值。
                    lanelines_pred, lanelines_prob = \
                        compute_3d_lanes_all_category(lane_anchors, dataset, args.anchor_y_steps, extrinsics[j][2,3])
                    json_line["laneLines"] = lanelines_pred
                    json_line["laneLines_prob"] = lanelines_prob
                    #json形式的数值保存到pred_lines_sub中。
                    pred_lines_sub.append(copy.deepcopy(json_line))

                    # # save 2d/3d eval results
                    # if args.evaluate:
                    #     img_path = json_line["file_path"]
                    #     self.save_eval_result(args, img_path, pred_decoded_2d[j], pred_decoded_2d_cate[j], lanelines_pred, lanelines_prob)


            eval_stats = self.evaluator.bench_one_submit_openlane_DDP(pred_lines_sub, gt_lines_sub, vis=False)

################################################################################################
            gather_output = [None for _ in range(args.world_size)]
            # all_gather all eval_stats and calculate mean
            dist.all_gather_object(gather_output, eval_stats)
            dist.barrier()
            r_lane = np.sum([eval_stats_sub[8] for eval_stats_sub in gather_output])
            p_lane = np.sum([eval_stats_sub[9] for eval_stats_sub in gather_output])
            c_lane = np.sum([eval_stats_sub[10] for eval_stats_sub in gather_output])
            cnt_gt = np.sum([eval_stats_sub[11] for eval_stats_sub in gather_output])
            cnt_pred = np.sum([eval_stats_sub[12] for eval_stats_sub in gather_output])
            match_num = np.sum([eval_stats_sub[13] for eval_stats_sub in gather_output])
            Recall = r_lane / (cnt_gt + 1e-6)
            Precision = p_lane / (cnt_pred + 1e-6)
            f1_score = 2 * Recall * Precision / (Recall + Precision + 1e-6)
            category_accuracy = c_lane / (match_num + 1e-6)

            eval_stats[0] = f1_score
            eval_stats[1] = Recall
            eval_stats[2] = Precision
            eval_stats[3] = category_accuracy
            eval_stats[4] = np.sum([eval_stats_sub[4] for eval_stats_sub in gather_output]) / args.world_size
            eval_stats[5] = np.sum([eval_stats_sub[5] for eval_stats_sub in gather_output]) / args.world_size
            eval_stats[6] = np.sum([eval_stats_sub[6] for eval_stats_sub in gather_output]) / args.world_size
            eval_stats[7] = np.sum([eval_stats_sub[7] for eval_stats_sub in gather_output]) / args.world_size

            return loss_list, eval_stats

