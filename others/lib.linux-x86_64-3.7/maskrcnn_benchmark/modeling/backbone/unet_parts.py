# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: unet_parts.py

# refer: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
#        https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    """
    (conv => BN => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels, padding=1, batch_norm=True):
        super(double_conv, self).__init__()
        block = []

        # conv 1
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        # conv 2

        block.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_channels))
        block.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class inconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(inconv, self).__init__()
        self.conv = double_conv(in_channels, out_channels, padding=1, batch_norm=True)

    def forward(self, x):
        x = self.conv(x)
        return x


class down_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down_sample, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels, padding=1, batch_norm=True)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_sample(nn.Module):
    def __init__(self, in_chanels, out_channels, padding, up_mode='upsample'):
        super(up_sample, self).__init__()

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_chanels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=2),
                                    nn.Conv2d(in_chanels, out_channels, kernel_size=1))

        self.conv_block = double_conv(in_chanels, out_channels, padding=padding, batch_norm=True)

    def forward(self, x1, x2):

        print('------X1-------')
        print('x1,', x1.shape)
        x1 = self.up(x1)

        print('------X1-------')
        print('x1,', x1.shape)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        print('------X1-------')
        print('x1,', x1.shape)

        print('------X2-------')
        print('x2,', x2.shape)

        out = torch.cat([x2, x1], dim=1)

        out = self.conv_block(out)

        return out


class outconv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
