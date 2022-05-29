# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: sca_attention2.py


import torch
import torch.nn as nn
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from maskrcnn_benchmark.utils.log_file_record import save_p_n_image_to_tensorboard, tb_log_prediction
from torchvision.utils import make_grid
from maskrcnn_benchmark.utils.log_file_record import tb_log_channel


# ------------------------------  Bot Up Use Channel-wise attention   -------------------------------------------------
# 一审投稿 NNN
# 通道注意力
class ChAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ChAttModule, self).__init__()

        # 池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.conv1x1(x)
        x = self.softmax(x)  # 使用 softmax

        return x


########################
# 在 bot up 结构使用 通道注意力机制
########################
class Botblock_CA(nn.Module):
    def __init__(self, in_planes, out_planes, ):
        super(Botblock_CA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        # 通道注意力
        self.channel_attention = ChAttModule(in_planes, out_planes)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add
        ca_feature = self.channel_attention(out)
        out = ca_feature * out + out

        return out


# ------------------------------  Bot Up Use Channel-wise attention   -------------------------------------------------


# ------------------------------  Bot Up Use Spatial-wise attention   -------------------------------------------------

# 针对意见 3 对比试验 1
# 空间注意力
class SAttModule(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SAttModule, self).__init__()

        # 降低为1通道
        self.conv_to_1 = nn.Sequential(
            nn.Conv2d(in_planes, 1, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv_to_1(x)

        x = self.softmax(x)
        return x


########################
# 在 bot up 结构使用 空间注意力机制 F * (1 + softmax)
########################
class Botblock_SA(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SA, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        # 空间注意力
        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add
        sa_feature = self.s_attention(out)
        out = sa_feature * out + out

        return out


# ------------------------------  Bot Up Use Channel-wise attention   -------------------------------------------------

class BotUpNet(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BotUpNet, self).__init__()

        # 仅使用空间注意力 F * (1 + softmax) ---> 意见3 对比实验1
        self.sa_block1 = Botblock_SA(in_planes, out_planes)
        self.sa_block2 = Botblock_SA(in_planes, out_planes)
        self.sa_block3 = Botblock_SA(in_planes, out_planes)

    def forward(self, p_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]

        N2 = P2
        N3 = self.sa_block1(N2, P3)
        N4 = self.sa_block2(N3, P4)
        N5 = self.sa_block3(N4, P5)

        return [N2, N3, N4, N5]


####################################################################
# 针对建议4的 对比实验9 FPN-SA2-- Bot-up-ca

class FPN_Botblock_SA2(nn.Module):
    """

    """

    def __init__(self, in_planes, out_planes, ):
        super(FPN_Botblock_SA2, self).__init__()

        # self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        # 通道注意力
        self.sa_attention = SAttModule(in_planes, 1)

    def forward(self, C, P):
        # 先对 P 上采样
        P = F.interpolate(P, scale_factor=2, mode='nearest')
        # C 和 P 相加
        c_p_add = C + P
        out = c_p_add
        # 对相加后的特征 使用空间注意力机制
        sa_feature = self.sa_attention(out)
        out = sa_feature * out

        return out


class FPN_SA2_Net(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FPN_SA2_Net, self).__init__()

        # 针对建议4, 对比实验2
        # 在 FPN 中使用空间注意力机制：
        # P_up --> C + P_up --> Stt(C+P_up)
        self.fpn_sa2_block1 = FPN_Botblock_SA2(in_planes, out_planes)
        self.fpn_sa2_block2 = FPN_Botblock_SA2(in_planes, out_planes)
        self.fpn_sa2_block3 = FPN_Botblock_SA2(in_planes, out_planes)

    def forward(self, cx_list):
        c5 = cx_list[3]
        c4 = cx_list[2]
        c3 = cx_list[1]
        c2 = cx_list[0]
        p5 = c5
        p4 = self.fpn_sa2_block1(c4, p5)
        p3 = self.fpn_sa2_block2(c3, p4)
        p2 = self.fpn_sa2_block3(c2, p3)

        return [p2, p3, p4, p5]


####################################################################


# 911 对比 FPN-SA2-CA + B-CA-SA2

class FPN_SA2_CA_block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FPN_SA2_CA_block, self).__init__()

        # 空间注意力机制
        self.sa_attention = SAttModule(in_planes, 1)

        # 通道注意力机制
        self.ca_attention = ChAttModule(in_planes, out_planes)

    def forward(self, C, P):
        # 先对 P 上采样
        P = F.interpolate(P, scale_factor=2, mode='nearest')
        # C 和 P 相加
        c_p_add = C + P
        out = c_p_add

        # 对相加后的特征 使用空间注意力机制
        sa_feature = self.sa_attention(out)
        out = sa_feature * out

        # 再使用通道注意力机制
        ca_out = self.ca_attention(out)
        out = ca_out * out + out

        return out


class FPN_SA2_CA_Net(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(FPN_SA2_CA_Net, self).__init__()
        self.sa2_ca_block1 = FPN_SA2_CA_block(in_planes, out_planes)
        self.sa2_ca_block2 = FPN_SA2_CA_block(in_planes, out_planes)
        self.sa2_ca_block3 = FPN_SA2_CA_block(in_planes, out_planes)

    def forward(self, cx_list):
        c5 = cx_list[3]
        c4 = cx_list[2]
        c3 = cx_list[1]
        c2 = cx_list[0]
        p5 = c5
        p4 = self.sa2_ca_block1(c4, p5)
        p3 = self.sa2_ca_block2(c3, p4)
        p2 = self.sa2_ca_block3(c2, p3)

        return [p2, p3, p4, p5]


class Botblock_CA_SA2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_CA_SA2, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        # 通道注意力机制
        self.ca_attention = ChAttModule(in_planes, out_planes)
        # 空间注意力
        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add

        # 先通道注意力机制
        ca_feature = self.ca_attention(out)
        out = out * ca_feature  # + out
        # 再空间注意力机制
        sa_feature = self.s_attention(out)
        out = sa_feature * out

        return out


class B_CA_SA2_Net(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(B_CA_SA2_Net, self).__init__()

        # 使用 通道-空间注意力机制
        self.ca_sa2_block1 = Botblock_CA_SA2(in_planes, out_planes)
        self.ca_sa2_block2 = Botblock_CA_SA2(in_planes, out_planes)
        self.ca_sa2_block3 = Botblock_CA_SA2(in_planes, out_planes)

    def forward(self, p_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]

        N2 = P2
        N3 = self.ca_sa2_block1(N2, P3)
        N4 = self.ca_sa2_block2(N3, P4)
        N5 = self.ca_sa2_block3(N4, P5)

        return [N2, N3, N4, N5]


class SE_Block(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SE_Block, self).__init__()
        self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features=in_planes, out_features=round(out_planes / 16))
        self.fc2 = nn.Linear(in_features=round(out_planes / 16), out_features=out_planes)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # print('---input size---: ', x.shape)
        origin_out = x
        out = self.globalAvgPool(x)
        # print('---out size---: ', out.shape)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = out.view(out.size(0), out.size(1), 1, 1)
        # out = self.relu(out * origin_out)

        return out


class Botblock_SE_SA2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Botblock_SE_SA2, self).__init__()

        self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)

        # 通道注意力机制
        self.ca_attention = SE_Block(in_planes, out_planes)
        # 空间注意力
        self.s_attention = SAttModule(in_planes, 1)

    def forward(self, N, P):
        x = self.conv3x3(N)
        p_n_add = x + P
        out = p_n_add

        # 先通道注意力机制
        ca_feature = self.ca_attention(out)
        out = out * ca_feature  # + out
        # 再空间注意力机制
        sa_feature = self.s_attention(out)
        out = sa_feature * out

        return out


class BotUpNet_SE_SA2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BotUpNet_SE_SA2, self).__init__()
        # self.p2_sa = P2Botblock(in_planes, out_planes)
        # self.botblock1 = Botblock(in_planes, out_planes)
        # self.botblock2 = Botblock(in_planes, out_planes)
        # self.botblock3 = Botblock(in_planes, out_planes)
        # self.botblock4 = Botblock(in_planes, out_planes)
        # self.conv3x3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        # self.conv1x1 = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv3 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 仅使用 通道注意力 softmax
        self.ca_block1 = Botblock_SE_SA2(in_planes, out_planes)
        self.ca_block2 = Botblock_SE_SA2(in_planes, out_planes)
        self.ca_block3 = Botblock_SE_SA2(in_planes, out_planes)
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
