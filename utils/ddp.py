
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import subprocess
import numpy as np
import random


#完成环境变量（WORLD_SIZE，RANK，LOCAL_RANK）的设置。
def setup_dist_launch(args):
    args.proc_id = args.local_rank
    world_size = torch.cuda.device_count()
    print("proc_id: " + str(args.proc_id))
    print("world size: " + str(world_size))
    print("local_rank: " + str(args.local_rank))

    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(args.proc_id)
    os.environ['LOCAL_RANK'] = str(args.local_rank)


def setup_distributed(args):

    # print("====enter setup_distributed.====")
    print(args.gpu)
    args.gpu = args.local_rank

    #step 1
    torch.cuda.set_device(args.gpu)
    #step 2
    # print(torch.distributed.is_nccl_available())

    dist.init_process_group(backend='nccl')
    args.world_size = dist.get_world_size()
    torch.set_printoptions(precision=10)

def ddp_init(args):
    args.proc_id, args.gpu, args.world_size = 0, 0, 1

    setup_dist_launch(args)

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) >= 1

    if args.distributed:
        setup_distributed(args)

    # deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #设置CPU生成随机数的种子，方便下次复现实验结果。
    torch.manual_seed(args.proc_id)
    np.random.seed(args.proc_id)
    random.seed(args.proc_id)

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


def reduce_tensors(*tensors, world_size):
    return [reduce_tensor(tensor, world_size) for tensor in tensors]