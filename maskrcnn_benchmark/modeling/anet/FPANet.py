


import torch
import torch.nn as nn
import torch.nn.functional as F


class FPC(nn.Module):
    def __init__(self, channels=256, mode='common'):
        super(FPC, self).__init__()

        # 256: 86 170
        # 256 降维到 86
        self.conv1x1_to_86 = nn.Conv2d(in_channels=channels, out_channels=86, kernel_size=1)

        # 256 降维到 170
        self.conv1x1_to_170 = nn.Conv2d(in_channels=channels, out_channels=170, kernel_size=1)

        # 256 降维到 128
        self.conv1x1_to_128 = nn.Conv2d(in_channels=channels, out_channels=128, kernel_size=1)

        # 512 降维到 256
        self.conv1x1_to_256 = nn.Conv2d(in_channels=2 * channels, out_channels=channels, kernel_size=1)

        # 缩小一半
        self.conv3x3_down = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)

    def forward(self, c_list):
        c5 = c_list[3]
        c4 = c_list[2]
        c3 = c_list[1]
        c2 = c_list[0]

        # cat(c4,c5) --> c5'
        c4_down = self.conv1x1_to_86(self.conv3x3_down(c4))
        c5_new = self.conv1x1_to_170(c5)
        c5_new = torch.cat([c5_new, c4_down], dim=1)

        # cat(c3,c2) --> c4'
        c5_new_up = self.conv1x1_to_128(F.interpolate(c5_new, scale_factor=2, mode='nearest'))
        c3_down = self.conv1x1_to_128(self.conv3x3_down(c3))
        c4_new = torch.cat([c5_new_up, c3_down], dim=1)

        # cat(c2,c4) --> c3'
        c4_new_up = self.conv1x1_to_128(F.interpolate(c4_new, scale_factor=2, mode='nearest'))
        c2_down = self.conv1x1_to_128(self.conv3x3_down(c2))
        c3_new = torch.cat([c4_new_up, c2_down], dim=1)

        # cat(c3,c2) --> c2'
        c3_new_up = self.conv1x1_to_86(F.interpolate(c3_new, scale_factor=2, mode='nearest'))
        c2_new = self.conv1x1_to_170(c2)
        c2_new = torch.cat([c3_new_up, c2_new], dim=1)

        return [c2_new, c3_new, c4_new, c5_new]


class FPN(nn.Module):
    def __init__(self, channels=256):
        super(FPN, self).__init__()

        self.cat1 = CAT(channels=channels)
        self.cat2 = CAT(channels=channels)

    def forward(self, c_new_list):
        c5 = c_new_list[3]
        c4 = c_new_list[2]
        c3 = c_new_list[1]
        c2 = c_new_list[0]

        p5 = c5

        # p4
        p5_up = F.interpolate(p5, scale_factor=2, mode='nearest')
        p4 = c4 + p5_up

        # p3
        p4_up = F.interpolate(p4, scale_factor=2, mode='nearest')
        p3 = c3 + p4_up

        # p2
        p3_up = F.interpolate(p3, scale_factor=2, mode='nearest')
        p2 = c2 + p3_up

        # cat(p5,p3)
        p4_new = self.cat1(p5, p3)
        p3_new = self.cat2(p4, p2)

        p3_new = F.interpolate(p4_new, scale_factor=2, mode='nearest') + p3_new

        p2_new = F.interpolate(p3_new, scale_factor=2, mode='nearest') + p2

        return [p2_new, p3_new, p4_new, p5]


class CAT(nn.Module):
    def __init__(self, channels):
        super(CAT, self).__init__()
        self.conv1x1_half_channel = nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=1)
        self.conv3x3_half_fms = nn.Conv2d(in_channels=channels, out_channels=channels // 2, kernel_size=3, stride=2,
                                          padding=1)

    def forward(self, high_fms, low_fms):
        high_fms = self.conv1x1_half_channel(high_fms)
        low_fms = self.conv3x3_half_fms(low_fms)

        high_fms_up = F.interpolate(high_fms, scale_factor=2, mode='nearest')

        out = torch.cat([high_fms_up, low_fms], dim=1)

        return out


class FPNCNet(nn.Module):
    def __init__(self, channels=256):
        super(FPNCNet, self).__init__()
        self.fpc = FPC(channels=256)
        self.fpn = FPN(channels=256)

    def forward(self, c_list):
        c_new_list = self.fpc(c_list)
        p_new_list = self.fpn(c_new_list)

        return p_new_list
