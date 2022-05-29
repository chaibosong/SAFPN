# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN

_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


# 该函数是创建模型的入口函数,也是唯一的模型创建函数
def build_detection_model(cfg):
    # 构建一个模型字典, 虽然只有一对键值, 但方便扩展
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    # 等价于 return GeneralizedRCNN(cfg) 根据配置信息创建模型
    return meta_arch(cfg)
