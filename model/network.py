import torch
import torchvision
import torch.nn as nn
import numpy as np
from efficientnet_pytorch import EfficientNet
from data.tools import gen_dx_bx, cumsum_trick, QuickCumsum

import torch.nn.functional as F


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
        # self.ref_resolution_layer = torch.log2(torch.tensor(downsample // 2 )).int()


        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        self.up1 = Up(320+112, 512) #b0 x16
        # self.up2 = Up(512+40, 256) #b0 x16

        # self.up1 = Up(320+40, 512, scale_factor=32/self.downsample) #b0 x8


        # self.up1 = Up(320+24, 512, scale_factor=32/self.downsample) #b0 x4
        
        # self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        
        # self.trunk = EfficientNet.from_pretrained("efficientnet-b6")
        # self.up1 = Up(576+200, 512)  #b6 x16
        # self.up1 = Up(576+72, 512, scale_factor=32/self.downsample) #b6 x8
        # self.up1 = Up(576+40, 512, scale_factor=32/self.downsample) #b6 x4


        # self.up1 = Up(640+224, 512) #b7



        # self.trunk = EfficientNet.from_pretrained("efficientnet-b5")
        # self.up1 = Up(512+176, 512) #b5 x 16


        # self.trunk = EfficientNet.from_pretrained("efficientnet-b3")
        # self.up1 = Up(384+136, 512) #b3 x 16




        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)
        # self.depthnet = nn.Conv2d(256, self.D + self.C, kernel_size=1, padding=0)


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
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4' ])
        # x = self.up2(x, endpoints['reduction_3'])

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
    def __init__(self, inC, stride=(2,1), padding=(1,1), kernel_size=3, offset_groups=1): 
        super(Deformable_conv, self).__init__() 
        self.stride = stride
        self.padding = padding
        self.kernel_size=kernel_size
        if inC % offset_groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        self.conv = nn.Conv2d(inC//offset_groups, inC , kernel_size=self.kernel_size, stride=self.stride,  padding=self.padding) #原卷积 
        # print(self.conv.weight.shape)
        self.conv_offset = nn.Conv2d(inC, offset_groups*2*self.kernel_size*self.kernel_size , kernel_size=self.kernel_size, stride=self.stride,  padding=self.padding) 
        init_offset = torch.Tensor(np.zeros([offset_groups*2*self.kernel_size*self.kernel_size, inC,  self.kernel_size, self.kernel_size])) 
        self.conv_offset.weight = torch.nn.Parameter(init_offset) #初始化为0 
        self.conv_mask = nn.Conv2d(inC, offset_groups*self.kernel_size*self.kernel_size , kernel_size=self.kernel_size, stride=self.stride,  padding=self.padding) 
        init_mask = torch.Tensor(np.zeros([offset_groups*self.kernel_size*self.kernel_size, inC, self.kernel_size, self.kernel_size])+np.array([0.5])) 
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
                                             mask=mask, padding=self.padding) 

        # print(out.shape)
        return out



class ResidualBlock(nn.Module):
    def __init__(self, inC, outC, stride=1):
        super(ResidualBlock, self).__init__()
        in_channels=inC
        out_channels=outC
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

# one block: Conv2d + (BN) + ReLu
def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False, inplace=True):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace)]
    else:
        layers = [conv2d, nn.ReLU(inplace)]
    return layers


class BEVConcatnate(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        num_outs):
    
        super(BEVConcatnate, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.up = nn.Upsample(scale_factor=2, mode='bilinear',
                    align_corners=True)
        self.lateral_convs = nn.ModuleList()


        self.lateral_convs = nn.ModuleList()
        self.regular_convs = nn.ModuleList()

        for i in range(0, self.num_ins):
            l_conv = make_one_layer(in_channels[i],
                                    out_channels,
                                    kernel_size=1,
                                    padding=0,
                                    batch_norm=True,
                                    inplace=False)
            self.lateral_convs.append(nn.Sequential(*l_conv))

            if (i != 0) :
                rc_conv = make_one_layer(out_channels*2,
                            out_channels,
                            kernel_size=1,
                            padding=0,
                            batch_norm=True,
                            inplace=False)
                self.regular_convs.append(nn.Sequential(*rc_conv))
           
        smooth_layer = make_one_layer(out_channels,
                                      out_channels,
                                      batch_norm=True,
                                      inplace=False)
        self.smooth_layer = (nn.Sequential(*smooth_layer))
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    print("=======conv weight=============")
                    nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    print("=======conv bias=============")
                    nn.init.constant_(m.bias, val=0)
            if isinstance(m, nn.Linear):
                if hasattr(m, 'weight') and m.weight is not None:
                    print("=======Linear weight=============")
                    nn.init.normal_(m.weight.data,0.1)
                if hasattr(m, 'bias') and m.bias is not None:
                    print("=======Linear bias=============")
                    nn.init.zeros_(m.bias.data)
            # if isinstance(m, nn.BatchNorm2d):
            #     print("=======BatchNorm2d bias=============")
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

		    # elif isinstance(m, nn.Linear):
			#     torch.nn.init.normal_(m.weight.data, 0.1)
			#     if m.bias is not None:
			# 	    torch.nn.init.zeros_(m.bias.data)
		    # elif isinstance(m, nn.BatchNorm2d):
			#     m.weight.data.fill_(1) 		 
			#     m.bias.data.zeros_()

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print("Max weight in fpn convs: ", torch.max(torch.abs(m.weight)))
                if m.weight.grad != None:
                    pass
                    # print("Max weight grad in fpn convs: ", torch.max(torch.abs(m.weight.grad)))
        

       # build laterals
            
        laterals = [lateral_conv(inputs[i])
                    for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # build top-down path
        for i in range( len(laterals) - 1, 0, -1):
            laterals[i - 1] = self.regular_convs[i - 1](torch.cat([laterals[i - 1],
                                                                    self.up(laterals[i])],dim=1))
        
        out = self.smooth_layer(laterals[0])

        return out



# modified from: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/necks/fpn.py

class FPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False):
        super(FPN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level  < inputs, no etra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs


        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = make_one_layer(in_channels[i],
                                    out_channels,
                                    kernel_size=1,
                                    padding=0,
                                    batch_norm=True if not self.no_norm_on_lateral else False,
                                    inplace=False)
            fpn_conv = make_one_layer(out_channels,
                                      out_channels,
                                      batch_norm=True,
                                      inplace=False)
            self.lateral_convs.append(nn.Sequential(*l_conv))
            self.fpn_convs.append(nn.Sequential(*fpn_conv))
        
        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = make_one_layer(in_channels,
                                                out_channels,
                                                kernel_size=3,
                                                stride=2,
                                                padding=1,
                                                batch_norm=True,
                                                inplace=False)
                self.fpn_convs.append(nn.Sequential(*extra_fpn_conv))
        # self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.xavier_uniform_(m.weight, gain=1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # print("Max weight in fpn convs: ", torch.max(torch.abs(m.weight)))
                if m.weight.grad != None:
                    pass
                    # print("Max weight grad in fpn convs: ", torch.max(torch.abs(m.weight.grad)))

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # laterals[i - 1] += self.up(laterals[i])
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1]  + F.interpolate(laterals[i], size=prev_shape, mode='nearest')
        
        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                else:
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        
        return outs[0]



# class BevEncode(nn.Module):
#     def __init__(self, inC):
#         super(BevEncode, self).__init__()

#         in_channels=[inC, inC*2, inC*4, inC*8]
#         self.fpn = FPN(in_channels, inC, num_outs= len(in_channels))
#         # self.fpn = BEVConcatnate(in_channels, inC, num_outs= len(in_channels))


#         self.layer1=ResidualBlock(inC=inC, outC=inC*2, stride=2)
#         self.layer2=ResidualBlock(inC=inC*2, outC=inC*4, stride=2)
#         self.layer3=ResidualBlock(inC=inC*4, outC=inC*8, stride=2)

    
#     def forward(self, x):
#         layer1= self.layer1(x)
#         layer2= self.layer2(layer1)
#         layer3= self.layer3(layer2)
#         x = self.fpn([x, layer1, layer2, layer3])
#         return x


class EfficientNetFPN(nn.Module):
    def __init__(self, inC):
        super(EfficientNetFPN, self).__init__()
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=inC)

        self.ref_resolution_layer = 4
        self.up1 = Up(320+112, 512) 
        self.up2 = Up(512 + 40, 256)
        self.up3 = Up(256 + 24, 256)
        self.up4 = Up(256 + 16, 64)

    def forward(self, x):
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
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        x = self.up2(x, endpoints['reduction_3']) 
        x= self.up3(x, endpoints['reduction_2']) 
        x= self.up4(x, endpoints['reduction_1']) 

        return x











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

    def __init__(self, args, grid_conf, data_aug_conf,   num_lane_type, num_y_steps, num_category):
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

        # self.bevencode = FPN(inC=self.camC)

        # self.bevencode = BevEncode(inC=self.camC)

        # self.dfc_3_all = Deformable_conv(inC=self.camC,  padding=(1,1), stride=(2,2),kernel_size=3)
        # self.dfc5_2 = Deformable_conv(inC=self.camC,  padding=(2,2), stride=(2,1),kernel_size=5)

        self.dfc3_2_layer1 = Deformable_conv(inC=self.camC)
        ## self.dropout1 = nn.Dropout(0.3) 
        self.dfc3_2_layer2 = Deformable_conv(inC=self.camC)
        ## self.dropout2 = nn.Dropout(0.3) 
        self.dfc3_2_layer3 = Deformable_conv(inC=self.camC)

        # self.dfc3_2 = Deformable_conv(inC=self.camC)


        # self.dfc = Deformable_conv(inC=self.camC,  padding=(1,1), stride=(1,1),kernel_size=3)


        self.head = LanePredictionHead(self.camC, num_lane_type, num_y_steps, num_category)
        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        # uncertainty loss weight
        self.uncertainty_loss = nn.Parameter(torch.tensor([args._3d_vis_loss_weight,
                                                            args._3d_prob_loss_weight,
                                                            args._3d_reg_loss_weight]), requires_grad=True)

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
        # x = self.bevencode(x)
        # x = self.dfc3_2(x)
        # x = self.dfc3_2(x)
        # x = self.dfc3_2(x)



        x = self.dfc3_2_layer1(x)
        ## x = self.dropout1(x) 
        x = self.dfc3_2_layer2(x)
        ## x = self.dropout2(x) 
        x = self.dfc3_2_layer3(x)



        # x= self.dfc_3_all(x)
        # x= self.dfc_3_all(x)
        # x= self.dfc_3_all(x)

        # x = self.bevencode(x)

        # x = self.dfc(x)
        # x = self.dfc5_2(x)
        x=self.head(x)

        uncertainty_loss = torch.tensor(1.0).to(x.device) * self.uncertainty_loss.to(x.device)

        return x , uncertainty_loss

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
def compile_model(args, grid_conf, data_aug_conf, num_lane_type, num_y_steps, num_category ):
    return LiftSplatShoot(args, grid_conf, data_aug_conf, num_lane_type, num_y_steps, num_category)





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

        self.vis_focal_gamma = vis_gamma = 2
        self.vis_focal_alpha = vis_alpha = 3

        self.category_focal_gamma = category_gamma = 3
        self.category_focal_alpha = category_alpha =2 


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
        pred_3D_lanes = pred_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)#self.num_types =1,   self.anchor_dim=51
        gt_3D_lanes = gt_3D_lanes.reshape(sizes[0], sizes[1], self.num_types, self.anchor_dim)
        # class prob N x ipm_w/8 x 3 x 1, anchor value N x ipm_w/8 x 3 x 2K
        pred_category = pred_3D_lanes[:, :, :, self.anchor_dim - self.num_category:]
        gt_category_onehot = gt_3D_lanes[:, :, :, self.anchor_dim - self.num_category:]

        pred_anchors = pred_3D_lanes[:, :, :, :2 * self.num_y_steps]
        pred_visibility = pred_3D_lanes[:, :, :, 2 * self.num_y_steps:3 * self.num_y_steps]
        gt_anchors = gt_3D_lanes[:, :, :, :2 * self.num_y_steps]
        gt_visibility = gt_3D_lanes[:, :, :, 2 * self.num_y_steps:3 * self.num_y_steps]

        # valid_category_weight = torch.sum(torch.mul(pred_category, gt_category_onehot), dim=-1)
        #valid_category_weight.shape =  (Batch,112,1,1)
        valid_category_weight = torch.sum(gt_category_onehot[:, :, :, 1:], dim=-1).unsqueeze(-1)

        # cross-entropy loss for visibility
        # eight * gt_vis * log(pred_vis)  +  weight * (1 - gt_vis)*(1 - pred_vis)
        vis_loss = -torch.sum(
            valid_category_weight * gt_visibility * torch.log(pred_visibility + torch.tensor(1e-9)) +
            valid_category_weight * (torch.ones_like(gt_visibility) - gt_visibility + torch.tensor(1e-9)) *
            torch.log(torch.ones_like(pred_visibility) - pred_visibility + torch.tensor(1e-9))) / self.num_y_steps

        pt = torch.exp(-vis_loss)
        loss0 = (1 - pt) ** self.vis_focal_gamma * vis_loss
        if self.vis_focal_alpha is not None:
            loss0 = self.vis_focal_alpha * loss0

        # balance categories
        num_category = pred_category.shape[-1]
        weight_category = 1.0 - torch.sum(gt_category_onehot.reshape(-1, num_category), dim=0) / torch.sum(
            gt_category_onehot)
        cross_entropy_loss = nn.CrossEntropyLoss(weight=weight_category, reduction='sum')
        # cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum')
        gt_category_onehot2class = torch.argmax(gt_category_onehot, dim=-1)
        pred_category = pred_category.reshape(-1, pred_category.shape[-1])
        gt_category_onehot2class = gt_category_onehot2class.reshape(-1)
        category_loss = cross_entropy_loss(pred_category, gt_category_onehot2class)

        pt = torch.exp(-category_loss)
        loss1 = (1 - pt) ** self.category_focal_gamma * category_loss
        if self.category_focal_alpha is not None:
            loss1 = self.category_focal_alpha * loss1


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

