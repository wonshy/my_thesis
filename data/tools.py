
import torch
import torch.optim
from torch.optim import lr_scheduler
import numpy as np
from scipy.interpolate import interp1d
import ortools
from ortools.graph import pywrapgraph


def projection_g2im_extrinsic(E, K):
    E_inv = np.linalg.inv(E)[0:3, :]
    P_g2im = np.matmul(K, E_inv)
    return P_g2im





def SolveMinCostFlow(adj_mat, cost_mat):
    """
        Solving an Assignment Problem with MinCostFlow"
    :param adj_mat: adjacency matrix with binary values indicating possible matchings between two sets
    :param cost_mat: cost matrix recording the matching cost of every possible pair of items from two sets
    :return:
    """

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    cnt_1, cnt_2 = adj_mat.shape
    cnt_nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    cnt_nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))

    # prepare directed graph for the flow
    start_nodes = np.zeros(cnt_1, dtype=np.int).tolist() +\
                  np.repeat(np.array(range(1, cnt_1+1)), cnt_2).tolist() + \
                  [i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1+1)] + \
                np.repeat(np.array([i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]).reshape([1, -1]), cnt_1, axis=0).flatten().tolist() + \
                [cnt_1 + cnt_2 + 1 for i in range(cnt_2)]
    capacities = np.ones(cnt_1, dtype=np.int).tolist() + adj_mat.flatten().astype(np.int).tolist() + np.ones(cnt_2, dtype=np.int).tolist()
    costs = (np.zeros(cnt_1, dtype=np.int).tolist() + cost_mat.flatten().astype(np.int).tolist() + np.zeros(cnt_2, dtype=np.int).tolist())
    # Define an array of supplies at each node.
    supplies = [min(cnt_nonzero_row, cnt_nonzero_col)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_nonzero_row, cnt_nonzero_col)]
    # supplies = [min(cnt_1, cnt_2)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_1, cnt_2)]
    source = 0
    sink = cnt_1 + cnt_2 + 1

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    match_results = []
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        # print('Total cost = ', min_cost_flow.OptimalCost())
        # print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    # print('set A item %d assigned to set B item %d.  Cost = %d' % (
                    #     min_cost_flow.Tail(arc)-1,
                    #     min_cost_flow.Head(arc)-cnt_1-1,
                    #     min_cost_flow.UnitCost(arc)))
                    match_results.append([min_cost_flow.Tail(arc)-1,
                                          min_cost_flow.Head(arc)-cnt_1-1,
                                          min_cost_flow.UnitCost(arc)])
    else:
        print('There was an issue with the min cost flow input.')

    return match_results




def SolveMinCostFlow(adj_mat, cost_mat):
    """
        Solving an Assignment Problem with MinCostFlow"
    :param adj_mat: adjacency matrix with binary values indicating possible matchings between two sets
    :param cost_mat: cost matrix recording the matching cost of every possible pair of items from two sets
    :return:
    """

    # Instantiate a SimpleMinCostFlow solver.
    min_cost_flow = pywrapgraph.SimpleMinCostFlow()
    # Define the directed graph for the flow.

    cnt_1, cnt_2 = adj_mat.shape
    cnt_nonzero_row = int(np.sum(np.sum(adj_mat, axis=1) > 0))
    cnt_nonzero_col = int(np.sum(np.sum(adj_mat, axis=0) > 0))

    # prepare directed graph for the flow
    start_nodes = np.zeros(cnt_1, dtype=np.int).tolist() +\
                  np.repeat(np.array(range(1, cnt_1+1)), cnt_2).tolist() + \
                  [i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]
    end_nodes = [i for i in range(1, cnt_1+1)] + \
                np.repeat(np.array([i for i in range(cnt_1+1, cnt_1 + cnt_2 + 1)]).reshape([1, -1]), cnt_1, axis=0).flatten().tolist() + \
                [cnt_1 + cnt_2 + 1 for i in range(cnt_2)]
    capacities = np.ones(cnt_1, dtype=np.int).tolist() + adj_mat.flatten().astype(np.int).tolist() + np.ones(cnt_2, dtype=np.int).tolist()
    costs = (np.zeros(cnt_1, dtype=np.int).tolist() + cost_mat.flatten().astype(np.int).tolist() + np.zeros(cnt_2, dtype=np.int).tolist())
    # Define an array of supplies at each node.
    supplies = [min(cnt_nonzero_row, cnt_nonzero_col)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_nonzero_row, cnt_nonzero_col)]
    # supplies = [min(cnt_1, cnt_2)] + np.zeros(cnt_1 + cnt_2, dtype=np.int).tolist() + [-min(cnt_1, cnt_2)]
    source = 0
    sink = cnt_1 + cnt_2 + 1

    # Add each arc.
    for i in range(len(start_nodes)):
        min_cost_flow.AddArcWithCapacityAndUnitCost(start_nodes[i], end_nodes[i],
                                                    capacities[i], costs[i])

    # Add node supplies.
    for i in range(len(supplies)):
        min_cost_flow.SetNodeSupply(i, supplies[i])

    match_results = []
    # Find the minimum cost flow between node 0 and node 10.
    if min_cost_flow.Solve() == min_cost_flow.OPTIMAL:
        # print('Total cost = ', min_cost_flow.OptimalCost())
        # print()
        for arc in range(min_cost_flow.NumArcs()):

            # Can ignore arcs leading out of source or into sink.
            if min_cost_flow.Tail(arc)!=source and min_cost_flow.Head(arc)!=sink:

                # Arcs in the solution have a flow value of 1. Their start and end nodes
                # give an assignment of worker to task.

                if min_cost_flow.Flow(arc) > 0:
                    # print('set A item %d assigned to set B item %d.  Cost = %d' % (
                    #     min_cost_flow.Tail(arc)-1,
                    #     min_cost_flow.Head(arc)-cnt_1-1,
                    #     min_cost_flow.UnitCost(arc)))
                    match_results.append([min_cost_flow.Tail(arc)-1,
                                          min_cost_flow.Head(arc)-cnt_1-1,
                                          min_cost_flow.UnitCost(arc)])
    else:
        print('There was an issue with the min cost flow input.')

    return match_results











def resample_laneline_in_y(input_lane, y_steps, out_vis=False):
    """
        Interpolate x, z values at each anchor grid, including those beyond the range of input lnae y range
    :param input_lane: N x 2 or N x 3 ndarray, one row for a point (x, y, z-optional).
                       It requires y values of input lane in ascending order
    :param y_steps: a vector of steps in y
    :param out_vis: whether to output visibility indicator which only depends on input y range
    :return:
    """

    # at least two points are included
    assert (input_lane.shape[0] >= 2)

    #为了差不在范围内的点，范围外（不是太大）的点可以采用。
    y_min = np.min(input_lane[:, 1]) - 5
    y_max = np.max(input_lane[:, 1]) + 5

    if input_lane.shape[1] < 3:
        input_lane = np.concatenate([input_lane, np.zeros([input_lane.shape[0], 1], dtype=np.float32)], axis=1)

    f_x = interp1d(input_lane[:, 1], input_lane[:, 0], fill_value="extrapolate")
    f_z = interp1d(input_lane[:, 1], input_lane[:, 2], fill_value="extrapolate")

    x_values = f_x(y_steps)
    z_values = f_z(y_steps)

    if out_vis:
        output_visibility = np.logical_and(y_steps >= y_min, y_steps <= y_max)
        return x_values, z_values, output_visibility.astype(np.float32) + 1e-9
    return x_values, z_values






def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx

def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1]) #相同的保留最右边的那个值， 注意这里是！ rank  ！

    x, geom_feats = x[kept], geom_feats[kept] #过滤出值来
    x = torch.cat((x[:1], x[1:] - x[:-1])) #错位加和后的值相减，得到加和操作。

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def define_optim(optim, params, lr, weight_decay):
    if optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'adamw':
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        optimizer = torch.optim.RMSprop(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("The requested optimizer: {} is not implemented".format(optim))
    return optimizer


def define_scheduler(optimizer, args):
    if args.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 - args.niter) / float(args.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=args.lr_decay_iters, gamma=args.gamma)
    elif args.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,
                                                   T_max=args.T_max, eta_min=args.eta_min)
    elif args.lr_policy == 'cosine_warm':
        '''
        lr_config = dict(
            policy='CosineAnnealing',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=1.0 / 3,
            min_lr_ratio=1e-3)
        '''
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                             T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min)
    elif args.lr_policy == 'None':
        scheduler = None
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)
    return scheduler

