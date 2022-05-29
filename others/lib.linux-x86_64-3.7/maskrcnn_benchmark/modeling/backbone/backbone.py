# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import unet


# 创建 resnet 骨架网络, 根据配置信息会被后面的 build_backbone() 函数调用
@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)  # resnet.py 文件中的 class ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))  # 利用 nn.Sequential 定义模型
    return model


# 创建 fpn 网络, 根据配置信息会被下面 build_backbone 函数调用
@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)  # 先创建 ResNet 网络提取特征

    # 获取 fpn 所需的channels参数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(  # 利用 fpn.py 文件夹的 class FPN 创建 fpn 网络
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


@registry.BACKBONES.register("UNet")
def build_unet_backbone(cfg):
    body = unet.UNet(cfg)  # resnet.py 文件中的 class ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))  # 利用 nn.Sequential 定义模型
    return model


@registry.BACKBONES.register("UNet-FPN")
def build_unet_fpn_backbone(cfg):
    body = unet.UNet(cfg)  # 先创建 ResNet 网络提取特征

    # 获取 fpn 所需的channels参数
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
    fpn = fpn_module.FPN(  # 利用 fpn.py 文件夹的 class FPN 创建 fpn 网络
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
