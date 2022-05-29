# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: attention_gate.py

import torch
import torch.nn.functional as F
from torch import nn


class AttentionGate(nn.Module):
    def __init__(self, mode='concatenation', dimension=2, sub_sample_factor=(2, 2, 2)):
        super(AttentionGate, self).__init__()
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.upsample_mode = 'nearest'

        self.theta = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2,
                               stride=2, padding=0, bias=False)

        self.phi = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=True)
        self.W = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0)

        if mode == 'concatenation':
            self.operation_function = self.attention_gate_block

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output

    def attention_gate_block(self, x, g):
        """

        :param x_y_add: Px 上采样后 和 Cx 相加
        :param out_channel:
        :return:
        """
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f_relu = F.relu(theta_x + phi_g, inplace=True)  # 激活 relu

        sigm_psi_f = F.sigmoid(self.psi(f_relu))
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode='nearest')
        y = sigm_psi_f.expand_as(x) * x
        w_y = self.W(y)
        result = w_y + x
        return result
