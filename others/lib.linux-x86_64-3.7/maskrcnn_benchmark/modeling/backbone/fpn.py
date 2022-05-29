# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
from .attention_gate import AttentionGate
from .bottom_up import BottomUp
from ..anet.FPANet import FPNCNet
from ..anet.sca_attention import FPAttention
from ..anet import sca_attention2, sca_attention


class FPN(nn.Module):
    """
    Module that adds FPN on top of a list of feature maps.
    The feature maps are currently supposed to be in increasing depth
    order, and must be consecutive
    """

    # 在一系列的 feature map (实际上就是stage2~5的最后一层输出)添加 FPN
    # 这些 feature maps 的 depth 假定是不断递增的, 并且 feature maps 必须是连续的(从stage角度)

    def __init__(
            self, in_channels_list, out_channels, conv_block, top_blocks=None
    ):
        """
        Arguments:
            in_channels_list (list[int]): number of channels for each feature map that
                will be fed
            out_channels (int): number of channels of the FPN representation
            top_blocks (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list
        """
        # in_channels_list (list[int]): 指示了送入 fpn 的每个 feature map 的通道数
        # out_channels (int): FPN表征的通道数, 所有的特征图谱最终都会转换成这个通道数大小
        # top_blocks (nn.Module or None): 当提供了 top_blocks 时, 就会在 FPN 的最后
        # 的输出上进行一个额外的 op, 然后 result 会扩展成 result list 返回

        super(FPN, self).__init__()

        # 创建两个空列表
        self.inner_blocks = []
        self.layer_blocks = []

        # 假设我们使用的是 ResNet-50-FPN 和配置, 则 in_channels_list 的值为:
        # [256, 512, 1024, 2048]

        for idx, in_channels in enumerate(in_channels_list, 1):  # 下标从1开始
            # 用下表起名: fpn_inner1, fpn_inner2, fpn_inner3, fpn_inner4
            inner_block = "fpn_inner{}".format(idx)

            # fpn_layer1, fpn_layer2, fpn_layer3, fpn_layer4
            layer_block = "fpn_layer{}".format(idx)

            # 创建 inner_block 模块, 这里 in_channels 为各个stage输出的通道数
            # out_channels 为 256, 定义在用户配置文件中
            # 这里的卷积核大小为1, 该卷积层主要作用为改变通道数到 out_channels(降维)
            inner_block_module = conv_block(in_channels, out_channels, 1)

            # 改变 channels 后, 在每一个 stage 的特征图谱上再进行 3×3 的卷积计算, 通道数不变
            layer_block_module = conv_block(out_channels, out_channels, 3, 1)

            # 在当前的特征图谱上添加 FPN
            self.add_module(inner_block, inner_block_module)
            self.add_module(layer_block, layer_block_module)

            # 将当前 stage 的 fpn 模块的名字添加到对应的列表当中
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

        # 将top_blocks作为 FPN 类的成员变量
        self.top_blocks = top_blocks

    def forward(self, x):
        """
        Arguments:
            x (list[Tensor]): feature maps for each feature level.
        Returns:
            results (tuple[Tensor]): feature maps after FPN layers.
                They are ordered from highest resolution first.
        """

        # x (list[Tensor]): 每个 feature level 的 feature maps,
        # ResNet的计算结果正好满足 FPN 的输入要求, 也因此可以使用 nn.Sequential 将二者直接结合
        # results (tuple[Tensor]): 经过FPN后的特征图谱组成的列表, 排列顺序是高分辨率的在前

        # 先计算最后一层(分辨率最低)特征图谱的fpn结果.
        last_inner = getattr(self, self.inner_blocks[-1])(x[-1])
        # print('-----------count----------', count)
        # print('layer_last shape: ', last_inner.shape)
        results = []

        # 将最后一层的计算结果添加到 results 中 C5 = P5 ??
        c5 = getattr(self, self.layer_blocks[-1])(last_inner)
        results.append(c5)
        cx_results = []
        cx_results.append(c5)

        # [:-1] 获取了前三项, [::-1] 代表从头到尾切片, 步长为-1, 效果为列表逆置
        # 举例来说, zip里的操作 self.inner_block[:-1][::-1] 的运行结果为
        # [fpn_inner3, fpn_inner2, fpn_inner1], 相当于对列表进行了逆置
        for feature, inner_block, layer_block in zip(
                x[:-1][::-1], self.inner_blocks[:-1][::-1], self.layer_blocks[:-1][::-1]
        ):
            # 根据给定的scale参数对特征图谱进行放大/缩小, 这里scale=2, 所以是放大
            # Px 上采样  ---> 2Px
            inner_top_down = F.interpolate(last_inner, scale_factor=2, mode="nearest")

            # 获取 inner_block 的计算结果
            # Cx 降维 ---> Cx
            inner_lateral = getattr(self, inner_block)(feature)
            cx_results.insert(0, inner_lateral)

            # TODO use size instead of scale to make it robust to different sizes
            # inner_top_down = F.upsample(last_inner, size=inner_lateral.shape[-2:],
            # mode='bilinear', align_corners=False)

            # 将二者叠加, 作为当前stage的输出 同时作为下一个stage的输入
            last_inner = inner_lateral + inner_top_down

            # 将当前stage输出添加到结果列表中, 注意还要用 layer_block 执行卷积计算
            # 同时为了使得分辨率最大的在前, 我们需要将结果插入到0位置
            results.insert(0, getattr(self, layer_block)(last_inner))

        # fpcn_net = FPNCNet(channels=256)
        # fpcn_net.cuda()
        # results_p = fpcn_net(cx_results)

        # fpn_attention = FPAttention(in_planes=256, out_planes=256)
        # fpn_attention.cuda()
        # a_results = fpn_attention(cx_results)
        # 建议4 对比1
        # bottom_up = BottomUp(channels=256, mode='common')
        # bottom_up.cuda()
        # results_new = bottom_up(results)
        # results = results_new

        # 建议4 对比 6 bot-up-ca
        bottom_up_ca = sca_attention2.BotUpNet_SE_SA2(256, 256)
        bottom_up_ca.cuda()
        results_new = bottom_up_ca(results)
        results = results_new

        #################################
        ## 建议4 对比 9 FPN-SA2 --- Bot_up_ca
        #################################
        # fpn_sa2 = sca_attention2.FPN_SA2_Net(256, 256)
        # fpn_sa2.cuda()
        # new_p_list = fpn_sa2(cx_results)
        #
        # bot_up_ca = sca_attention.BotUpNet(256, 256)
        # bot_up_ca.cuda()
        # new_results = bot_up_ca(new_p_list)
        #
        # results = new_results

        # bot_up 通达注意力
        # ca_bot_up = sca_attention2.BotUpNet(in_planes=256, out_planes=256)
        # ca_bot_up.cuda()
        # new_results = ca_bot_up(results)
        # results = new_results
        # fpn
        # fpn_att = sca_attention.FPAttentionMoudle(in_planes=256, out_planes=256)
        # fpn_att.cuda()
        # new_results = fpn_att(cx_results)
        # results = new_results
        #
        # # bottom-up
        # bottom_up = BottomUp(256, 'common')
        # bottom_up.cuda()
        # results_new = bottom_up(results)
        # results = results_new

        # fpn_sa2_ca = sca_attention2.FPN_SA2_Net(256, 256)
        # fpn_sa2_ca.cuda()
        # p_list = fpn_sa2_ca(cx_results)

        # # 对比 911: FPN-SA2-CA + B-CA-SA2
        #
        # fpn_sa2_ca = sca_attention2.FPN_SA2_CA_Net(256, 256)
        # fpn_sa2_ca.cuda()
        # p_list = fpn_sa2_ca(cx_results)

        # b_ca_sa2 = sca_attention2.B_CA_SA2_Net(256, 256)
        # b_ca_sa2.cuda()
        # new_results = b_ca_sa2(p_list)

        # results = new_results

        # 如果 top_blocks 不为空, 则执行这些额外op
        if self.top_blocks is not None:
            last_results = self.top_blocks(results[-1])
            results.extend(last_results)

        # 以元组(只读)形式返回
        return tuple(results)


# 最后一级的 max pool 层
class LastLevelMaxPool(nn.Module):
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]
