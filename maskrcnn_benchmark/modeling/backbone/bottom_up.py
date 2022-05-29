# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: bottom_up.py

import torch
from torch import nn


# 参考 PANet
class BottomUp(nn.Module):
    def __init__(self, channels=256, mode='common'):
        super(BottomUp, self).__init__()
        # 卷积变为原来的一半
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=2, padding=1)
        self.conv1x1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(channels)

        if mode == 'common':
            self.operation_function = self.bot_up_common

    def forward(self, p_list):
        return self.operation_function(p_list)

    def bot_up_common(self, p_list):
        P5 = p_list[3]
        P4 = p_list[2]
        P3 = p_list[1]
        P2 = p_list[0]

        N2 = P2
        # N3 = self.conv1(N2) + P3
        # N4 = self.conv1(N3) + P4
        # N5 = self.conv1(N4) + P5
        N3 = self.relu(self.conv2(self.conv1(N2) + P3))
        N4 = self.relu(self.conv2(self.conv1(N3) + P4))
        N5 = self.relu(self.conv2(self.conv1(N4) + P5))

        return [N2, N3, N4, N5]
