
import torch.nn as nn
import torch
from torch.nn import init
from ..anet import sca_attention2
import torch.nn.functional as F
from ..anet.utils.utils_extra import Conv2dStaticSamePadding, MaxPool2dStaticSamePadding
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from ..anet.sca_attention2 import Botblock_CA_SA2

class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, return_nl_map=False):
        """
        :param x: (b, c, t, h, w)
        :param return_nl_map: if True return z, nl_map, else only return z.
        :return:
        """

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        if return_nl_map:
            return z, f_div_C
        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SeparableConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, norm=True, activation=False, onnx_export=False):
        super(SeparableConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        # Q: whether separate conv
        #  share bias between depthwise_conv and pointwise_conv
        #  or just pointwise_conv apply bias.
        # A: Confirmed, just pointwise_conv applies bias, depthwise_conv has no bias.

        self.depthwise_conv = Conv2dStaticSamePadding(in_channels, in_channels,
                                                      kernel_size=3, stride=1, groups=in_channels, bias=False)
        self.pointwise_conv = Conv2dStaticSamePadding(in_channels, out_channels, kernel_size=1, stride=1)

        self.norm = norm
        if self.norm:
            # Warning: pytorch momentum is different from tensorflow's, momentum_pytorch = 1 - momentum_tensorflow
            self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)

        self.activation = activation

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.norm:
            x = self.bn(x)
        # if self.activation:
        #     x = self.swish(x)

        return x

class SAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAttModule, self).__init__()
        self.conv_to_1 = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_to_1(x)

        x = self.softmax(x)
        return x

class ChAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChAttModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)  

        return x
class Botblock_CA_SA3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_CA_SA3, self).__init__()

        self.channel_attention = ChAttModule(in_planes, out_planes)
        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, x):

        out = x

        ca_feature = self.channel_attention(out)
        out = ca_feature * out + out

        sa_feature = self.s_attention(out)
        out = sa_feature * out 
        return out


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        gate_channels = [channel]
        gate_channels += [channel // reduction] * num_layers
        gate_channels += [channel]

        self.ca = nn.Sequential()
        self.ca.add_module('flatten', Flatten())
        for i in range(len(gate_channels) - 2):
            self.ca.add_module('fc%d' % i, nn.Linear(gate_channels[i], gate_channels[i + 1]))
            self.ca.add_module('bn%d' % i, nn.BatchNorm1d(gate_channels[i + 1]))
            self.ca.add_module('relu%d' % i, nn.ReLU())
        self.ca.add_module('last_fc', nn.Linear(gate_channels[-2], gate_channels[-1]))

    def forward(self, x):
        res = self.avgpool(x)
        res = self.ca(res)
        res = res.unsqueeze(-1).unsqueeze(-1).expand_as(x)
        return res


class SpatialAttention(nn.Module):
    def __init__(self, channel, reduction=16, num_layers=3, dia_val=2):
        super().__init__()
        self.sa = nn.Sequential()
        self.sa.add_module('conv_reduce1',
                           nn.Conv2d(kernel_size=1, in_channels=channel, out_channels=channel // reduction))
        self.sa.add_module('bn_reduce1', nn.BatchNorm2d(channel // reduction))
        self.sa.add_module('relu_reduce1', nn.ReLU())
        for i in range(num_layers):
            self.sa.add_module('conv_%d' % i, nn.Conv2d(kernel_size=3, in_channels=channel // reduction,
                                                        out_channels=channel // reduction, padding=1, dilation=dia_val))
            self.sa.add_module('bn_%d' % i, nn.BatchNorm2d(channel // reduction))
            self.sa.add_module('relu_%d' % i, nn.ReLU())
        self.sa.add_module('last_conv', nn.Conv2d(channel // reduction, 1, kernel_size=1))

    def forward(self, x):
        res = self.sa(x)
        res = res.expand_as(x)
        return res

class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True,
                 use_p8=False):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon
        self.use_p8 = use_p8


        self.conv_block=conv_with_kaiming_uniform()

        self.conv3x3_2 = self.conv_block(256, 256, 3, 1)
        self.conv3x3_3 = self.conv_block(256, 256, 3, 1)
        self.conv3x3_4 = self.conv_block(256, 256, 3, 1)
        self.conv3x3_5 = self.conv_block(256, 256, 3, 1)

        self.conv1x1_2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_4 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1_5 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0, bias=False)


        self.bn2 = nn.BatchNorm2d(num_features=256, momentum=0.01, eps=1e-3)
        self.bn3 = nn.BatchNorm2d(num_features=256, momentum=0.01, eps=1e-3)
        self.bn4 = nn.BatchNorm2d(num_features=256, momentum=0.01, eps=1e-3)
        self.bn5 = nn.BatchNorm2d(num_features=256, momentum=0.01, eps=1e-3)

        self.swish_2 = Swish()
        self.swish_3 = Swish()
        self.swish_4 = Swish()
        self.swish_5 = Swish()

        self.attention_11 = Botblock_CA_SA3(256, 256)
        self.attention_22 = Botblock_CA_SA3(256, 256)
        self.attention_33 = Botblock_CA_SA3(256, 256)
        self.attention_44 = Botblock_CA_SA3(256, 256)

     
        self.first_time=first_time
        if self.first_time:
            self.c5_down_channel=Conv2dStaticSamePadding(conv_channels[3], num_channels, 1)
            self.c4_down_channel=Conv2dStaticSamePadding(conv_channels[2], num_channels, 1)
            self.c3_down_channel=Conv2dStaticSamePadding(conv_channels[1], num_channels, 1)
            self.c2_down_channel=Conv2dStaticSamePadding(conv_channels[0], num_channels, 1)
        


    def forward(self, inputs):
        if self.first_time:
            c2, c3, c4, c5 = inputs
            
            c2 = self.c2_down_channel(c2)
            c3 = self.c3_down_channel(c3)
            c4 = self.c4_down_channel(c4)
            c5 = self.c5_down_channel(c5)
            p5 = c5
            
            p5_upsample = F.interpolate(p5, scale_factor=2, mode="nearest")
            p4 = c4 + p5_upsample
            p4_upsample = F.interpolate(p4, scale_factor=2, mode="nearest")
            p3 = c3 + p4_upsample
            p3_upsample = F.interpolate(p3, scale_factor=2, mode="nearest")
            p2 = c2 + p3_upsample
        else:
            c2, c3, c4, c5 = inputs
            p5 = self.conv1x1_5(c5)
            p5_upsample = F.interpolate(p5, scale_factor=2, mode="nearest")
            p4 =self.conv1x1_4(c4)+p5_upsample
            p4_upsample = F.interpolate(p4, scale_factor=2, mode="nearest")
            p3 = self.conv1x1_3(c3)+p4_upsample
            p3_upsample = F.interpolate(p3, scale_factor=2, mode="nearest")
            p2 =self.conv1x1_2(c2) + p3_upsample

        out2 = self.swish_2(self.conv3x3_2(p2) + c2)
        out3 = self.attention_11(self.swish_3(self.conv3x3_3(p3) + c3 + F.interpolate(out2, scale_factor=0.5, mode="nearest")))
        out4 = self.attention_22(self.swish_4(self.conv3x3_4(p4) + c4 + F.interpolate(out3, scale_factor=0.5, mode="nearest")))
        out5 = self.attention_33(self.swish_5(self.conv3x3_5(p5) + c5 + F.interpolate(out4, scale_factor=0.5, mode="nearest")))

  

        return [out2, out3, out4, out5]

