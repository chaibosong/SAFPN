# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.

    # 该类是 MaskrcnnBenchmark 中所有模型的共同抽象, 目前支持 boxes 和 masks 两种形式的标签
    # 该类主要包含以下三个部分:
    # - backbone
    # - rpn(option)
    # - heads: 利用前面网络输出的 features 和 proposals 来计算 detections / masks.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        # 根据配置信息创建 backbone 网络
        self.backbone = build_backbone(cfg)  # 特征提取的 backbone 网络

        # 根据配置信息创建 RPN 网络
        self.rpn = build_rpn(cfg)  # 生成区域建议

        # 根据配置信息创建 roi_heads
        self.roi_heads = build_roi_heads(cfg)  # 对 ROI 进行 bbox 回归和 mask 预测

    def forward(self, images, targets=None):  # 定义模型的前向传播过程
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        # 当 training 为 True 时， 必须提供 targets
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)  # 把图片数据类型转换成 ImageList

        # 利用 backbone 网络 获取图片的 features
        # print('--------IMAGES----------')
        # print(images.tensors)
        # print('--------------------')
        features = self.backbone(images.tensors)

        # 利用 RPN 网络 获取 proposals 和 相应的 loss
        proposals, proposal_losses = self.rpn(images, features, targets)
        if self.roi_heads:  # 如何 roi_heads 不为 None 的话, 就计算其输出的结果
            x, result, detector_losses = self.roi_heads(features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:  # 训练模式下, 输出损失值, 更新loss
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses

        return result  # 非训练模式, 则输出模型预测结果
