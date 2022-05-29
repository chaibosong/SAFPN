# -*- coding: utf-8 -*-
# @Author  : MengYangD
# @FileName: unet.py


# refer:
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
# https://github.com/usuyama/pytorch-unet/blob/master/pytorch_unet.py
# https://github.com/jvanvugt/pytorch-unet/blob/master/unet.py


import torch.nn as nn
import torch.nn.functional as F
from .unet_parts import *
from maskrcnn_benchmark.utils.registry import Registry
from maskrcnn_benchmark.layers import FrozenBatchNorm2d


class BaseStem(nn.Module):
    def __init__(self, cfg, norm_func):
        super(BaseStem, self).__init__()

        out_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.conv1 = nn.Conv2d(
            3, out_channels, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_func(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu_(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class StemWithFixedBatchNorm(BaseStem):
    def __init__(self, cfg):
        super(StemWithFixedBatchNorm, self).__init__(
            cfg, norm_func=FrozenBatchNorm2d
        )


class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        # Translate string names to implementations
        stem_module = _STEM_MODULES[cfg.MODEL.RESNETS.STEM_FUNC]
        self.stem = stem_module(cfg)
        in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS

        self.inc = inconv(in_channels, 64)
        self.down1 = down_sample(64, 128)
        self.down2 = down_sample(128, 256)
        self.down3 = down_sample(256, 512)
        self.down4 = down_sample(512, 1024)

        self.up1 = up_sample(1024, 512, padding=1, up_mode='upsample')
        self.up2 = up_sample(512, 256, padding=1, up_mode='upsample')
        self.up3 = up_sample(256, 128, padding=1, up_mode='upsample')
        self.up4 = up_sample(128, 64, padding=1, up_mode='upsample')

        # self.outc = outconv(64,2)

    def forward(self, x):
        # x1 = self.inc(x)
        x1 = self.stem(x)
        print('------------')
        print('x1,', x1.shape)
        x2 = self.inc(x1)
        print('------------')
        print('x2,', x2.shape)
        down1 = self.down1(x2)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        print('------------')
        print('d1,', down1.shape)
        print('------------')
        print('d2,', down2.shape)
        print('------------')
        print('d3,', down3.shape)
        print('------------')
        print('d4,', down4.shape)

        up1 = self.up1(down4, down3)
        up2 = self.up2(up1, down2)
        up3 = self.up3(up2, down1)
        up4 = self.up4(up3, x1)
        # out = self.outc(up4)

        outputs_res = [up4, up2, up3, up1]

        # return F.sigmoid(out)
        return outputs_res


_STAGE_SPECS = Registry({
    "UNet-FPN": UNet,
})

_STEM_MODULES = Registry({
    "StemWithFixedBatchNorm": StemWithFixedBatchNorm,
})
