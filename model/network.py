import torch
import torchvision
import torch.nn as nn
import numpy as np
from efficientnet_pytorch import EfficientNet
from data.tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

    
        self.downsample = downsample

        # self.ref_resolution_layer = (torch.tensor(5) - torch.log2(torch.tensor(32/(self.downsample)))).int()
        self.ref_resolution_layer = torch.log2(torch.tensor(self.downsample)).int()


        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, 512) #b0 x16
        # self.up1 = Up(320+24, 512, scale_factor=32/self.downsample) #b0 x4
        
        # self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        # self.up1 = Up(576+200, 512)  #b6 x16
        # self.up1 = Up(576+72, 512, scale_factor=32/self.downsample) #b6 x8
        # self.up1 = Up(576+40, 512, scale_factor=32/self.downsample) #b6 x4


        # self.up1 = Up(640+224, 512) #b7

        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        k1 = depth.unsqueeze(1)
        k2 = x[:, self.D:(self.D + self.C)].unsqueeze(2)

        new_x = k1 * k2

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_{:d}'.format(self.ref_resolution_layer) ])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


##### normal

# class BevEncode(nn.Module):
#     def __init__(self, inC):
#         super(BevEncode, self).__init__()
#         reduce_layers = []
#         extract_layers = []

#         dilation = nn.Conv2d(inC, inC, kernel_size=3, padding=(2,2), dilation=(2,2),stride=1) 
#         conv2d = nn.Conv2d(inC, inC, kernel_size=3, padding=(1,1),stride=1) 
#         reduce_layers += [dilation, nn.BatchNorm2d(inC), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)] 
#         extract_layers += [conv2d, nn.BatchNorm2d(inC), nn.ReLU(inplace=True)] 

#         # layers += [conv2d, nn.BatchNorm2d(inC), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)] 

#         self.reduce_features = nn.Sequential(*reduce_layers)
#         self.extruct_features = nn.Sequential(*extract_layers)


#         # self.dilation = nn.Conv2d(inC, inC, kernel_size=3, padding=(1,1), stride=1)

#     def forward(self, x):
#         x = self.reduce_features(x)
#         # x = self.extruct_features(x)
#         # x = self.reduce_features(x)


#         # x = self.extruct_features(x)
#         return x


### deformable conv 
class BevEncode(nn.Module): 
    def __init__(self, inC): 
        super(BevEncode, self).__init__() 
        self.offset_group = 64
        if inC % self.offset_group != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.conv = nn.Conv2d(inC//self.offset_group, inC , kernel_size=3, stride=(2,2),  padding=1) #原卷积 
        print(self.conv.weight.shape)
        self.conv_offset = nn.Conv2d(inC, self.offset_group*2*3*3 , kernel_size=3, stride=(2,2),  padding=1) 
        init_offset = torch.Tensor(np.zeros([self.offset_group*2*3*3, inC,  3, 3])) 
        self.conv_offset.weight = torch.nn.Parameter(init_offset) #初始化为0 
        self.conv_mask = nn.Conv2d(inC, self.offset_group*3*3 , kernel_size=3, stride=(2,2),  padding=1) 
        init_mask = torch.Tensor(np.zeros([self.offset_group*3*3, inC, 3, 3])+np.array([0.5])) 
        self.conv_mask.weight = torch.nn.Parameter(init_mask) #初始化为0.5
    def forward(self, x): 
        offset = self.conv_offset(x) 
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间 
        # input (Tensor[batch_size, in_channels, in_height, in_width]) – input tensor
        # offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]) – offsets to be applied for each position in the convolution kernel.
        # weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]) – convolution weights, split into groups of size (in_channels // groups)
        # bias (Tensor[out_channels]) – optional bias of shape (out_channels,). Default: None
        # stride (int or Tuple[int, int]) – distance between convolution centers. Default: 1
        # padding (int or Tuple[int, int]) – height/width of padding of zeroes around each image. Default: 0
        # dilation (int or Tuple[int, int]) – the spacing between kernel elements. Default: 1
        # mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]) – masks to be applied for each position in the convolution kernel. Default: None
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, stride=(2,2),
                                            weight=self.conv.weight,  
                                             mask=mask, padding=(1, 1)) 
        return out





# ############  Residual Network
# class BevEncode(nn.Module):
#     def __init__(self, inC, stride=2):
#         super(BevEncode, self).__init__()
#         self.conv1 = nn.Conv2d(inC, inC, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(inC)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(inC, inC, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(inC)
        
#         if stride != 1 or inC != inC:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(inC, inC, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(inC)
#             )
#         else:
#             self.downsample = None
    
#     def forward(self, x):
#         identity = x
        
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
        
#         out = self.conv2(out)
#         out = self.bn2(out)
        
#         if self.downsample is not None:
#             identity = self.downsample(x)
        
#         out += identity
#         out = self.relu(out)
        
#         return out


class Deformable_conv(nn.Module): 
    def __init__(self, inC, stride, offset_groups): 
        super(Deformable_conv, self).__init__() 
        self.stride = stride
        if inC % offset_groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.conv = nn.Conv2d(inC//offset_groups, inC , kernel_size=3, stride=self.stride,  padding=1) #原卷积 
        print(self.conv.weight.shape)
        self.conv_offset = nn.Conv2d(inC, offset_groups*2*3*3 , kernel_size=3, stride=self.stride,  padding=1) 
        init_offset = torch.Tensor(np.zeros([offset_groups*2*3*3, inC,  3, 3])) 
        self.conv_offset.weight = torch.nn.Parameter(init_offset) #初始化为0 
        self.conv_mask = nn.Conv2d(inC, offset_groups*3*3 , kernel_size=3, stride=self.stride,  padding=1) 
        init_mask = torch.Tensor(np.zeros([offset_groups*3*3, inC, 3, 3])+np.array([0.5])) 
        self.conv_mask.weight = torch.nn.Parameter(init_mask) #初始化为0.5
    def forward(self, x): 
        offset = self.conv_offset(x) 
        mask = torch.sigmoid(self.conv_mask(x)) #保证在0到1之间 
        # input (Tensor[batch_size, in_channels, in_height, in_width]) – input tensor
        # offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width, out_height, out_width]) – offsets to be applied for each position in the convolution kernel.
        # weight (Tensor[out_channels, in_channels // groups, kernel_height, kernel_width]) – convolution weights, split into groups of size (in_channels // groups)
        # bias (Tensor[out_channels]) – optional bias of shape (out_channels,). Default: None
        # stride (int or Tuple[int, int]) – distance between convolution centers. Default: 1
        # padding (int or Tuple[int, int]) – height/width of padding of zeroes around each image. Default: 0
        # dilation (int or Tuple[int, int]) – the spacing between kernel elements. Default: 1
        # mask (Tensor[batch_size, offset_groups * kernel_height * kernel_width, out_height, out_width]) – masks to be applied for each position in the convolution kernel. Default: None
        out = torchvision.ops.deform_conv2d(input=x, offset=offset, stride=self.stride,
                                            weight=self.conv.weight,  
                                             mask=mask, padding=(1, 1)) 
        return out



class ResidualBlock(nn.Module):
    def __init__(self, inC, stride=1):
        super(ResidualBlock, self).__init__()
        in_channels=inC
        out_channels=inC
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out









#  Lane Prediction Head: through a series of convolutions with no padding in the y dimension, the feature maps are
#  reduced in height, and finally the prediction layer size is N × 1 × 3 ·(3 · K + 1)
class LanePredictionHead(nn.Module):
    def __init__(self, input_channels, num_lane_type, num_y_steps, num_category, batch_norm=True):
        super(LanePredictionHead, self).__init__()
        self.num_lane_type = num_lane_type
        self.num_y_steps = num_y_steps

        # self.anchor_dim = 3*self.num_y_steps + 1
        self.anchor_dim = 3 * self.num_y_steps + num_category  # (x, z, vis) + category conf
        self.num_category = num_category

        layers = []
        layers += make_one_layer(input_channels, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=3, padding=(0, 1), batch_norm=batch_norm)

        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        layers += make_one_layer(64, 64, kernel_size=5, padding=(0, 2), batch_norm=batch_norm)
        self.features = nn.Sequential(*layers)

        # x suppose to be N X 64 X 4 X ipm_w/8, need to be reshaped to N X 256 X ipm_w/8 X 1
        # TODO: use large kernel_size in x or fc layer to estimate z with global parallelism
        dim_rt_layers = []
        dim_rt_layers += make_one_layer(256, 128, kernel_size=(5, 1), padding=(2, 0), batch_norm=batch_norm)
        dim_rt_layers += [nn.Conv2d(128, self.num_lane_type * self.anchor_dim, kernel_size=(5, 1), padding=(2, 0))]
        self.dim_rt = nn.Sequential(*dim_rt_layers)

    def forward(self, x):

        x = self.features(x)
        # x suppose to be N X 64 X 4 X ipm_w/8, reshape to N X 256 X ipm_w/8 X 1
        # TODO: this only works with fpn_out_channels=128 & num_proj=4
        sizes = x.shape
        # print("x shape: ", x.size())
        x = x.reshape(sizes[0], sizes[1] * sizes[2], sizes[3], 1)
        x = self.dim_rt(x)
        x = x.squeeze(-1).transpose(1, 2)

        # apply sigmoid to the visbility terms to make it in (0, 1)
        for i in range(self.num_lane_type):
            # x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim] = \
            #     torch.sigmoid(x[:, :, i*self.anchor_dim + 2*self.num_y_steps:(i+1)*self.anchor_dim])
            x[:, :, i * self.anchor_dim + 2 * self.num_y_steps:i * self.anchor_dim + 3 * self.num_y_steps] = \
                torch.sigmoid(
                    x[:, :, i * self.anchor_dim + 2 * self.num_y_steps:i * self.anchor_dim + 3 * self.num_y_steps])
            # x[:, :, i*self.anchor_dim + 2*self.num_y_steps : i*self.anchor_dim + 3*self.num_y_steps] = \
            #     torch.sigmoid(
            #         x[:, :, i*self.anchor_dim + 2*self.num_y_steps : i*self.anchor_dim + 3*self.num_y_steps])
        return x



class LiftSplatShoot(nn.Module):

    def __init__(self, grid_conf, data_aug_conf,   num_lane_type, num_y_steps, num_category):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC)
        self.head = LanePredictionHead(self.camC, num_lane_type, num_y_steps, num_category)
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        #print(frustum)

        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points
    def my_get_geometry(self, rots, trans, intrins):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape

        brodcast = torch.zeros(N*B*3, device=trans.device).reshape(B, N, 1, 1, 1, 3)

        points = self.frustum - brodcast
        points = points.unsqueeze(-1)
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))

        #TODO， data type  need change.
        combine = combine.float()
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points




    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W  # C:64, H:8, N:5, D:41, B:1 W:32

        # flatten x
        x = x.reshape(Nprime, C)  # 展开成模式（8×5×41×1×31， 64）

        # flatten indices
        # 这里完成的是 geom_feats 减去  (xbound, ybound, zbound)第一个元素 ， 之后再除以 (xbound, ybound, zbound)最后一个元素, 最后再投射成为整形。 (geom_feats -（-50,-50,-10）)/ (0.5, 0.5, 20)
        # 最终的目的是：完成geom_feats数据转换到BEV的平面
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)  # 展开成模式（8×5×41×1×31， 3）
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in
                              range(B)])  # 8×5×41×1×31 个index值。比如index=0,则8×5×41×1×31个0, index=1, 8×5×41×1×31个1
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # 添加一个维度，在geom_feats后面，即所属batch的数字。

        # filter out points that are outside box
        # 过滤范围的数据为（0-200,0-220,0-1），将符合范围的数据留下来。
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]  # 筛选出来

        # get tensors from the same voxel next to each other
        # 从彼此相邻的同一体素中获取张量，
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x Y x X)
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final


    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x = self.get_cam_feats(x)
    
        x = self.voxel_pooling(geom, x)
    
        return x
    
    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        # x = self.bevencode(x)

        x=self.head(x)
        return x

#     # nn.MaxPool2d(kernel_size=2, stride=2)
# #x dim (bach size, camera num, channel, height, width)
#     def my_get_voxels(self,x, rots, trans, intrins):
#         geom = self.my_get_geometry(rots, trans, intrins)
#         x = self.get_cam_feats(x)

#         x = self.voxel_pooling(geom, x)

#         return x
    # def forward(self, x, rots, trans, intrins):
    #     x = self.my_get_voxels(x, rots, trans, intrins)
    #     x = self.bevencode(x)
    #     x=self.head(x)
    #     # print("==============")
    #     return x


#TODO: outC  is camera channel?
def compile_model(grid_conf, data_aug_conf, num_lane_type, num_y_steps, num_category ):
    return LiftSplatShoot(grid_conf, data_aug_conf, num_lane_type, num_y_steps, num_category)



# one block: Conv2d + (BN) + ReLu
def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False, inplace=True):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace)]
    else:
        layers = [conv2d, nn.ReLU(inplace)]
    return layers

#we need loss0 , loss1 , loss2, pred_cam=False
class Laneline_loss_gflat_multiclass(nn.Module):
    """
    Compute the loss between predicted lanelines and ground-truth laneline in anchor representation.
    The anchor representation is in flat ground space X', Y' and real 3D Z. Visibility estimation is also included.

    loss = loss0 + loss1 + loss2 + loss2
    loss0: cross entropy loss for lane point visibility
    loss1: cross entropy loss for lane type classification
    loss2: sum of geometric distance betwen 3D lane anchor points in X and Z offsets
    loss3: error in estimating pitch and camera heights
    """
    def __init__(self, num_types, num_y_steps, pred_cam, num_category, loss_dist):
        super(Laneline_loss_gflat_multiclass, self).__init__()
        self.num_types = num_types
        self.num_y_steps = num_y_steps
        self.num_category = num_category
        self.loss_dist = loss_dist      # a list, weight for loss0, loss1, loss2, loss3

        self.anchor_dim = 3* self.num_y_steps + num_category
        self.pred_cam = pred_cam

    def forward(self, pred_3D_lanes, gt_3D_lanes):
        """

        :param pred_3D_lanes: predicted tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param gt_3D_lanes: ground-truth tensor with size N x (ipm_w/8) x 3*(2*K+1)
        :param pred_pitch: predicted pitch with size N
        :param gt_pitch: ground-truth pitch with size N
        :param pred_hcam: predicted camera height with size N
        :param gt_hcam: ground-truth camera height with size N
        :return:
        """
        sizes = pred_3D_lanes.shape
        # reshape to N x ipm_w/8 x 3 x (3K+1)
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_category = pred_3D_lanes[:, :, :, self.anchor_dim - self.num_category:]
        gt_category_onehot = gt_3D_lanes[:, :, :, self.anchor_dim - self.num_category:]

        pred_anchors = pred_3D_lanes[:, :, :, :2 * self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2 * self.num_y_steps:3 * self.num_y_steps]
        gt_anchors = gt_3D_lanes[:, :, :, :2 * self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2 * self.num_y_steps:3 * self.num_y_steps]

        # valid_category_weight = torch.sum(torch.mul(pred_category, gt_category_onehot), dim=-1)
        valid_category_weight = torch.sum(gt_category_onehot[:, :, :, 1:], dim=-1).unsqueeze(-1)

        # cross-entropy loss for visibility

        loss0 = -torch.sum(
            valid_category_weight * gt_visibility * torch.log(pred_visibility + torch.tensor(1e-9)) +
            valid_category_weight * (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9))) / self.num_y_steps

        # balance categories
        num_category = pred_category.shape[-1]
        weight_category = 1.0 - torch.sum(gt_category_onehot.reshape(-1, num_category), dim=0) / torch.sum(
            gt_category_onehot)
        cross_entropy_loss = nn.CrossEntropyLoss(weight=weight_category, reduction='sum')
        # cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        gt_category_onehot2class = torch.argmax(gt_category_onehot, dim=-1)
        pred_category = pred_category.reshape(-1, pred_category.shape[-1])
        gt_category_onehot2class = gt_category_onehot2class.reshape(-1)
        loss1 = cross_entropy_loss(pred_category, gt_category_onehot2class)


        # x/z offsets
        loss2 = torch.sum(torch.norm(valid_category_weight * torch.cat((gt_visibility, gt_visibility), 3) *
                                     (pred_anchors - gt_anchors), p=1, dim=3))

        # Batch mean
        # loss0 /= sizes[0]
        # loss1 /= sizes[0]
        # loss2 /= sizes[0]
        # loss0 = self.loss_dist[0] * loss0
        # loss1 = self.loss_dist[1] * loss1
        # loss2 = self.loss_dist[2] * loss2



        return self.loss_dist[0] * loss0 + self.loss_dist[1] * loss1 + self.loss_dist[2] * loss2, {
            'vis_loss': loss0, 'prob_loss': loss1, 'reg_loss': loss2}

