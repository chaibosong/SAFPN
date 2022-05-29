# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: sca_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from maskrcnn_benchmark.utils.log_file_record import save_p_n_image_to_tensorboard, tb_log_prediction
from torchvision.utils import make_grid
from maskrcnn_benchmark.utils.log_file_record import tb_log_channel


class SAModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAModule, self).__init__()

        # 降低为1通道
        self.conv_to_one = nn.Sequential(
            nn.Conv2d(in_planes, out_planes // 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.Conv2d(out_planes // 2, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d()
        )
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_to_one(x)

        x = self.sigmoid(x)
        # x = x / x.sum()
        return x


class CAModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CAModule, self).__init__()

        # 池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.global_pool(x)
        # print('------------')
        # print('pool0', x.shape)
        x = self.conv1x1(x)
        x = self.sigmoid(x)
        # x = x / x.sum()
        # print('------------')
        # print('pool', x.shape)

        return x


class SCANet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SCANet, self).__init__()

        # 池化
        self.sa = SAModule(in_planes, out_planes)
        # self.ca = CAModule(in_planes, out_planes)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # print('----------xshape')
        # print('x shape', x.shape)
        # ac = self.ca(x)
        # print('----------ac shape')
        # print('ac shape', ac.shape)
        # f_ca = x * ac + x
        # f_ca = self.conv1x1(f_ca)
        a_s = self.sa(x)

        # print('f_ca shape', f_ca.shape)
        # print('f_ca shape', f_ca.shape)
        f_cs = a_s * x
        f_cs = self.conv1x1(f_cs)
        return f_cs


class FPAttention(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FPAttention, self).__init__()

        # 池化
        self.sca1 = SCANet(in_planes, out_planes)
        self.sca2 = SCANet(in_planes, out_planes)
        self.sca3 = SCANet(in_planes, out_planes)
        self.sca4 = SCANet(in_planes, out_planes)

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, c_list):
        c5 = c_list[3]
        c4 = c_list[2]
        c3 = c_list[1]
        c2 = c_list[0]

        # print('c5 shape', c5.shape)
        # print('c4 shape', c4.shape)
        # print('c3 shape', c3.shape)
        # print('c2 shape', c2.shape)

        s5 = self.sca1(c5) + c5
        s5_2 = F.interpolate(s5, scale_factor=2, mode='nearest')
        # s5_2 = self.conv3x3(torch.cat([s5_2, c4], dim=1))  # + c4
        # s5_2 = self.conv3x3(s5_2)

        s4 = self.sca2(s5_2) + c4
        s4_2 = F.interpolate(s4, scale_factor=2, mode='nearest')
        # s4_2 = self.conv3x3(torch.cat([s4_2, c3], dim=1))  # + c3
        # s4_2 = self.conv3x3(s4_2)  # + c3

        s3 = self.sca2(s4_2) + c3
        s3_2 = F.interpolate(s3, scale_factor=2, mode='nearest')
        # s3_2 = self.conv3x3(torch.cat([s3_2, c2], dim=1))  # + c2
        # s3_2 = self.conv3x3(s3_2)

        s2 = self.sca3(s3_2) + c2

        # print('s5 shape', s5.shape)
        # print('s4 shape', s4.shape)
        # print('s3 shape', s3.shape)
        # print('s2 shape', s2.shape)

        return [s2, s3, s4, s5]


class SAModule2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAModule2, self).__init__()

        # 降低为1通道
        self.conv_to_one = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d()
        )
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_to_one(x)
        x = self.sigmoid(x)
        return x


class SAModule3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAModule3, self).__init__()

        # 降低为1通道
        self.conv_to_half = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 2, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d()

        )
        self.bn = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_to_half(x)
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.sigmoid(x) + x
        return x


class CAModule2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CAModule2, self).__init__()

        # 池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv1x1(x)
        x = self.sigmoid(x)

        return x


class CAModule3(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(CAModule3, self).__init__()

        # 池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv1x1(x)
        x = self.softmax(x) + x

        return x


# 通道注意力
class ChAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChAttModule, self).__init__()

        # 池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)  # 使用 softmax

        return x


# 空间注意力
class SAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAttModule, self).__init__()

        # 降低为1通道
        self.conv_to_1 = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.conv1x1 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        # self.bn = nn.BatchNorm2d(1)
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_to_1(x)
        x = self.conv1x1(x)
        x = self.softmax(x)
        return x


########################
# 在 bot up 结构使用 通道注意力机制
########################
class Botblock_CA(nn.Module):
    def __init__(self, in_planes, out_planes, ):
        super(Botblock_CA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 通道注意力
        self.channel_attention = ChAttModule(in_planes, out_planes)

        # self.relu = nn.ReLU(inplace=True)
        # self.bn = nn.BatchNorm2d(out_planes)
        # self.nx = n_x

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P

        out = p_n_add
        ca_feature = self.channel_attention(out)

        out = ca_feature * out  # + out

        return out


########################
# 在 bot up 结构使用 空间注意力机制
########################
class Botblock_SA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 通道注意力
        # self.channel_attention = ChAttModule(in_planes, out_planes)

        # 空间注意力
        self.s_attention = SAttModule(in_planes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add
        sa_feature = self.s_attention(out)
        out = sa_feature * out + out
        return out


###########################
# 在 bot up 结构使用 空间和通道注意力机制
###########################

class Botblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 通道注意力
        # self.ca2 = CAModule2(in_planes, out_planes)
        # self.ca3 = CAModule3(in_planes, out_planes)
        # 空间注意力
        self.sa3 = SAModule3(in_planes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = self.conv1x1(x + P)
        # ca = self.ca2(p_n_add)
        # out = ca * p_n_add + p_n_add
        out = p_n_add
        sa = self.sa3(out)
        out = self.conv1x1(sa * out + out)
        return self.bn(self.relu(out))


########################
# 在 bot up 结构使用 空间 通道 注意力机制
########################
class Botblock_SA_CA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SA_CA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 通道注意力
        self.channel_attention = ChAttModule(in_planes, out_planes)

        # 空间注意力
        self.s_attention = SAttModule(in_planes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add

        # 空间注意力机制
        sa_feature = self.s_attention(out)
        out = self.bn(self.relu(sa_feature * out + out))

        # 通道注意力机制
        ca_feature = self.channel_attention(out)
        out = self.bn(self.relu(ca_feature * out + out))
        return out


########################
# 在 bot up 结构使用 空间-通道 注意力机制
########################
class Botblock_CA_SA2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_CA_SA2, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 通道注意力
        self.channel_attention = ChAttModule(in_planes, out_planes)

        # 空间注意力
        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add

        # 通道注意力机制
        ca_feature = self.channel_attention(out)
        ca_out = ca_feature * out + out

        # 空间注意力机制
        sa_feature = self.s_attention(ca_out)
        out = sa_feature * ca_out  # + ca_out

        return out


class P2Botblock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(P2Botblock, self).__init__()

        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 空间注意力
        self.sa3 = SAModule3(in_planes, 1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, P2):
        sa = self.sa3(P2)
        out = P2
        out = self.conv1x1(sa * out + out)
        return self.bn(self.relu(out))


class BotUpNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BotUpNet, self).__init__()
        # self.p2_sa = P2Botblock(in_planes, out_planes)
        # self.botblock1 = Botblock(in_planes, out_planes)
        # self.botblock2 = Botblock(in_planes, out_planes)
        # self.botblock3 = Botblock(in_planes, out_planes)
        # self.botblock4 = Botblock(in_planes, out_planes)
        # self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 仅使用 通道注意力 softmax
        self.ca_block1 = Botblock_CA_SA2(in_planes, out_planes)
        self.ca_block2 = Botblock_CA_SA2(in_planes, out_planes)
        self.ca_block3 = Botblock_CA_SA2(in_planes, out_planes)
        # 仅使用空间注意力 softmax
        # self.sa_block1 = Botblock_SA(in_planes, out_planes)
        # self.sa_block2 = Botblock_SA(in_planes, out_planes)
        # self.sa_block3 = Botblock_SA(in_planes, out_planes)

        # 空间,通道注意力机制
        # self.sa_ca_block1 = Botblock_SA_CA(in_planes, out_planes)
        # self.sa_ca_block2 = Botblock_SA_CA(in_planes, out_planes)
        # self.sa_ca_block3 = Botblock_SA_CA(in_planes, out_planes)

    def forward(self, p_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]

        # N2 = self.p2_sa(P2)
        N2 = P2
        N3 = self.ca_block1(N2, P3)
        N4 = self.ca_block2(N3, P4)
        N5 = self.ca_block3(N4, P5)
        # N5 = self.sa_block3(N4, P5)
        # N5 = self.conv3(self.conv3(N5) + P5)
        # N5 = self.conv1x1(self.conv3x3(N4) + P5)

        # 先绘制 p2~p5
        # p_list = [P2, P3, P4, P5]
        # save_image_to_tensorboard('p', feature_map_list=p_list)

        # n_list = [N3, N4, N5]
        # save_image_to_tensorboard('n', feature_map_list=n_list)
        # save_p_n_image_to_tensorboard(p_feature_map_list=p_list, n_feature_map_list=n_list)
        # tb_log_prediction(p_list, n_list)

        return [N2, N3, N4, N5]


# class FPAMoudle(nn.Module):
#     def __init__(self, in_planes, out_planes):
#         super(FPAMoudle, self).__init__()
#
################################# FPN

########################
# 在 bot up 结构使用 通道注意力机制
########################
class Botblock_CA_F(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_CA_F, self).__init__()

        # self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)

        # 通道注意力
        self.channel_attention = ChAttModule(in_planes, out_planes)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, C, P):
        # x = self.conv3x3(N)
        p_up = F.interpolate(P, scale_factor=2, mode='nearest')
        p_n_add = self.conv1x1(C + p_up)
        out = p_n_add
        ca_feature = self.channel_attention(out)

        out = self.conv1x1(ca_feature * out + out)

        return self.bn(self.relu(out))


class FPAttentionMoudle(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FPAttentionMoudle, self).__init__()
        self.ca4 = Botblock_CA_F(in_planes, out_planes)
        self.ca3 = Botblock_CA_F(in_planes, out_planes)
        self.ca2 = Botblock_CA_F(in_planes, out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, c_list):
        c5 = c_list[3]
        c4 = c_list[2]
        c3 = c_list[1]
        c2 = c_list[0]

        p5 = self.bn(self.relu(c5))
        p4 = self.ca4(c4, p5)
        p3 = self.ca3(c3, p4)
        p2 = self.ca2(c2, p3)
        return [p2, p3, p4, p5]
